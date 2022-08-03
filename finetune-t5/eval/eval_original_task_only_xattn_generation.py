import argparse
import logging
import os
import random
import json
from re import template
from glob import glob
import datasets
import torch
from datasets import (
    load_dataset,
    load_metric,
    load_from_disk,
    concatenate_datasets
)
import datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    default_data_collator,
)
from promptsource.templates import DatasetTemplates

from caliberation import calibrate_probs, get_task_template_name, load_template_dict
# use custom data collator to add indices
import sys
src_root = os.path.dirname(os.getcwd())
TRAINING_DIR = os.path.join(src_root, 'training') # TODO
sys.path.insert(1, TRAINING_DIR)
from data_collator import DataCollatorForMultipleChoice, DataCollatorForMultipleChoiceXattn, DataCollatorForSeq2SeqXattnODQA
from modeling_t5 import (
    T5ForConditionalGenerationWithPerceiverResamplerXattnOnEncoder, 
    T5ForConditionalGenerationWithPerceiverResamplerXattnOnDecoder, 
    T5ForConditionalGenerationSharedEncoderXattnOnDecoder,
    T5ForConditionalGenerationMultiAug,
    T5ForConditionalGenerationMultiAug_Base,
    T5ForConditionalGenerationMultiAug_DoubleGated
)
logger = logging.getLogger(__name__)

def get_dataset_name_2_if_origianl():
    task_2_templates = json.load(open("/data1/mikeeewang/data/bigscience_P3_task_2_templates.json"))
    dataset_name_2_if_origianl = {}
    for key,value in task_2_templates.items():
        if 'original_dataset_name' in value:
            for d in value['original_dataset_name']:
                dataset_name_2_if_origianl[d] = True
        if 'omit_dataset_name' in value:
            for d in value['omit_dataset_name']:
                dataset_name_2_if_origianl[d] = False
    return dataset_name_2_if_origianl

####################################################
######## evaluation for open domain qa #############
####################################################

#Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
import regex
import string
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions (strings) to score.
    references: list of possible answers for each prediction. Each
        reference should be a list of strings.
Returns:
    exact_match: exact_match score
"""
class ODQAMetric(datasets.Metric):
    """metric for open domain qa evaluation;
    returns:
        exact_match
    """

    def _info(self):
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description="metric for open domain qa evaluation",
            citation="",
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': [datasets.Value('string')],
            }),
            # # Homepage of the metric for documentation
            # homepage="http://metric.homepage",
            # # Additional links to the codebase or references
            # codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            # reference_urls=["http://path.to.reference.url/new_metric"]
        )

    # def _download_and_prepare(self, dl_manager):
    #     """Optional: download external resources useful to compute the scores"""
    #     # TODO: Download external resources if needed
    #     bad_words_path = dl_manager.download_and_extract(BAD_WORDS_URL)
    #     self.bad_words = {w.strip() for w in open(bad_words_path, encoding="utf-8")}

    def _compute(self, predictions, references):
        """Returns the scores"""
        em_results = []
        for i in range(len(predictions)):
            em_result = ems(predictions[i], references[i])
            em_results.append(em_result)
        
        return {
            "exact_match": (sum(em_results)/len(em_results))*100
        }


def unit_test():
    metric = ODQAMetric()
    predicts = ["cat", "dog"]
    gts = [["cat "],["dog1","the dog"]]
    metric.add_batch(predictions=predicts, references=gts)

    predicts = ["aaa", "ccc"]
    gts = [["bbb"],["ddd"]]
    metric.add_batch(predictions=predicts, references=gts)
    print(metric.compute())


### arg options ###
def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")
    parser.add_argument(
        "--model_architecture",
        type=str,
        required=True,
        help=(
            "chosen from ['fusionOnEncoder', 'fusionOnDecoder']"
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--aug_max_length",
        type=int,
        default=512,
        help=(
            "augmentation max length"
        ),
    )
    parser.add_argument(
        "--target_max_length",
        type=int,
        default=256,
        help="Target max length. Sequences longer than this will be truncated."
    )
    parser.add_argument(
        "-ftd",
        "--filter_dataset",
        action="store_true",
        help="if doing filtering (filter out non-original) on datasets based on the template name",
    )
    parser.add_argument(
        "-cia",
        "--concat_input_at_augmentation",
        action="store_true",
        help=(
            "If passed, concat input at each augmentation"
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--saved_model_step",
        type=int,
        default=None,
        help=(
            "load saved model checkpoint step"
        ),
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--error_analysis_dir",
        type=str,
        default="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/error_analysis",
        help="Where to store the error analysis files"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--parallelize",
        action="store_true",
        help=(
            "If passed, will call `model.parallelize` which splits the model on all GPUs available when applicable (model parallelism). "
            "Note that this feature is still experimental in HF Transformers."
        ),
    )
    parser.add_argument(
        "-pdp",
        "--processed_dataset_paths",
        type=str,
        nargs='+',
        default=[],
        help=""
    )
    parser.add_argument(
        "--eval_average",
        action="store_true",
        help="If passed, will only evaluate average score.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="If passed, use calibration before use.",
    )

    args = parser.parse_args()

    return args

### main ###
def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # max length config
    logger.info(f"input max_length: {args.max_length}")
    logger.info(f"augmentation max_length: {args.aug_max_length}")
    logger.info(f"target max_length: {args.target_max_length}")

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the output directory creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    assert os.path.isdir(args.model_name_or_path)
    args_config = json.load(open(os.path.join(args.model_name_or_path, "args.json")))
    perceiver_config = json.load(open(os.path.join(args.model_name_or_path, "perceiver_config.json")))

    config = AutoConfig.from_pretrained(args_config['lm_name'])
    
    logger.info(config)
    logger.info(perceiver_config)

    if args.tokenizer_name:
        logger.info(f'tokenizer: {args.tokenizer_name}')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    else:
        logger.info(f'tokenizer:')
        logger.info(args_config['lm_name'])
        tokenizer = AutoTokenizer.from_pretrained(args_config['lm_name'], use_fast=not args.use_slow_tokenizer)

    if tokenizer.pad_token is None:
        for token in [tokenizer.eos_token, tokenizer.bos_token, tokenizer.sep_token]:
            if token is not None:
                tokenizer.pad_token = token
        if tokenizer.pad_token is None:
            raise ValueError("Please define a pad token id.")

    logger.info(f"model architecture: {args.model_architecture}")
    if args.model_architecture == 'fusionOnDecoder':
        model_class_name = T5ForConditionalGenerationWithPerceiverResamplerXattnOnDecoder
    elif args.model_architecture == 'fusionOnEncoder':
        model_class_name = T5ForConditionalGenerationWithPerceiverResamplerXattnOnEncoder
    elif args.model_architecture == 'SharedEncoder_fusionOnDecoder':
        model_class_name = T5ForConditionalGenerationSharedEncoderXattnOnDecoder
    elif args.model_architecture == 'SharedEncoderDecoder_MultiAug':
        model_class_name = T5ForConditionalGenerationMultiAug
    elif args.model_architecture == 'SharedEncoderDecoder_MultiAug_Base':
        model_class_name = T5ForConditionalGenerationMultiAug_Base
    elif args.model_architecture == 'SharedEncoderDecoder_MultiAug_DoubleGated':
        model_class_name = T5ForConditionalGenerationMultiAug_DoubleGated
    else:
        raise NotImplementedError
    model = model_class_name(
        config,
        perceiver_xattn_config = perceiver_config,
        # freeze_lm = True,
        # cross_attn_every=args_config['cross_attn_every'],
        # only_attend_immediate_media=False,
        # num_xattn_layers=args_config['args_config']
    )

    assert args.saved_model_step is not None
    if args.saved_model_step == -1:
        ckpt_paths = sorted(glob(os.path.join(args.model_name_or_path, f"step_*")))
    else:
        ckpt_paths = [os.path.join(args.model_name_or_path, f"step_{args.saved_model_step}")]

    logger.info("eval steps:")
    logger.info(ckpt_paths)
    
    MAX_AUG_NUM = perceiver_config["num_aug_sources"]
    logger.info(f"MAX_AUG_NUM:{MAX_AUG_NUM}")
    logger.info(f"if concat input text at each augmentation: {args.concat_input_at_augmentation}")

    ### main loop over step checkpoints
    for ckpt_path in ckpt_paths:

        logger.info(f"loading ckpt from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        # Preprocessing the datasets.
        padding = "max_length" if args.pad_to_max_length else False

        if args.calibrate:
            # load template look up table
            template_dict = load_template_dict()

        def process_eval(examples, indices):
            bs = len(examples['inputs_pretokenized'])

            input_texts = [] # list of strings
            target_texts = [] # list of strings
            augmentation_texts_list = [] # list of list of strings
            answer_texts_list = []
            for i in range(bs):
                ex = {
                    k: examples[k][i] for k in ["inputs_pretokenized", "chosen_examples", "targets_pretokenized", "answers"]
                }

                input_text = ex['inputs_pretokenized'].strip()
                target_text = ex['targets_pretokenized'].strip()
                assert isinstance(ex['chosen_examples'],list)
                
                augmentation_texts = []
                for item in ex['chosen_examples'][:MAX_AUG_NUM]:
                    aug = item["inputs_pretokenized"].strip()
                    if len(aug) < 2:
                        logger.info("WARNING: fall back to default aug: N/A")
                        aug = "N/A"
                    if args.concat_input_at_augmentation:
                        aug = input_text + "\n" + aug
                    augmentation_texts.append(aug)

                input_texts.append(input_text)
                target_texts.append(target_text)
                augmentation_texts_list.append(augmentation_texts)
                answer_texts_list.append(ex['answers'])

            tokenized_inputs = tokenizer(
                input_texts,
                padding=padding,
                max_length=args.max_length,
                truncation=True,
                add_special_tokens=False
            )

            tokenized_augmentations = []
            for i in range(bs):
                tokenized_augmentations_per_instance = tokenizer(
                    augmentation_texts_list[i],
                    padding=padding,
                    max_length=args.aug_max_length,
                    truncation=True,
                    add_special_tokens=False
                )
                tokenized_augmentations.append(tokenized_augmentations_per_instance)

            tokenized_answers = []
            for i in range(bs):
                tokenized_answers_per_instance = tokenizer(
                    answer_texts_list[i],
                    padding=padding,
                    max_length=args.target_max_length,
                    truncation=True,
                    add_special_tokens=False
                )
                tokenized_answers.append(tokenized_answers_per_instance)

            tokenized_targets = tokenizer(
                target_texts,
                padding=padding,
                max_length=args.max_length,
                truncation=True,
                add_special_tokens=True
            )

            features = {
                "input_ids":tokenized_inputs.input_ids,
                "aug_input_ids":[item.input_ids for item in tokenized_augmentations],
                "attention_mask":tokenized_inputs.attention_mask,
                "aug_attention_mask":[item.attention_mask for item in tokenized_augmentations],
                "labels":tokenized_targets.input_ids,
                "answer_input_ids":[item.input_ids for item in tokenized_answers]
            }
            return features


        # mapping to check if original task
        if args.filter_dataset:
            dataset_name_2_if_origianl = get_dataset_name_2_if_origianl()

        raw_eval_datasets = []
        for dataset_path in args.processed_dataset_paths:
            dataset = os.path.basename(dataset_path)

            if args.filter_dataset:
                if dataset not in dataset_name_2_if_origianl:
                    logger.info('ERROR: unseen dataset:',dataset)
                    quit()
                if not dataset_name_2_if_origianl[dataset]:
                    logger.info(f'!!! skip non-original:{dataset}')
                    continue

            if '_score_eval' in dataset:
                continue
            logger.info(f'loading dataset: {dataset}')
            raw_dataset = load_from_disk(dataset_path)
            try:
                raw_eval_dataset = raw_dataset['validation']
            except KeyError:
                logger.warning(f'no validation set, skip {dataset}')
                continue
            
            raw_eval_datasets.append(raw_eval_dataset)
            
            # if 'answer_choices' in raw_eval_dataset.features:
            #     raw_eval_datasets.append(raw_eval_dataset)
            # else:
            #     logger.warning(f'no `answer_choices`, skip {dataset}')

        if args.eval_average:
            raw_eval_datasets = [concatenate_datasets(raw_eval_datasets)]
            args.processed_dataset_paths = ['average']

        for dataset_path, raw_eval_dataset in zip(args.processed_dataset_paths, raw_eval_datasets):
            column_names = raw_eval_dataset.column_names if raw_eval_dataset else None
            
            if args.calibrate:
                task_full_name, template_name = get_task_template_name(os.path.dirname(dataset_path), dataset_path, wic_aug=None)

            eval_dataset = raw_eval_dataset.map(
                process_eval,
                batched=True,
                remove_columns=column_names,
                with_indices=True
            )
            # # Log a few random samples from the eval set:
            # for index in random.sample(range(len(eval_dataset)), 3):
            #     logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

            # DataLoaders creation:
            if args.pad_to_max_length:
                # If padding was already done ot max length, we use the default data collator that will just convert everything
                # to tensors.
                data_collator = default_data_collator
            else:
                # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
                # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
                # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
                data_collator = DataCollatorForSeq2SeqXattnODQA(
                    tokenizer,
                    model=model,
                    label_pad_token_id=-100,
                    pad_to_multiple_of=8 if accelerator.use_fp16 else None
                )
                # data_collator = DataCollatorForMultipleChoiceXattn(
                #     tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
                # )
                # data_collator = DataCollatorForMultipleChoice(
                #     tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
                # )

            eval_dataloader = DataLoader(
                eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


            # # Use the device given by the `accelerator` object.
            if not args.parallelize:
                model.to(accelerator.device)

            # Prepare everything with our `accelerator`.
            eval_dataloader = accelerator.prepare(eval_dataloader)

            # Metrics
            # metric = load_metric("exact_match")
            metric = ODQAMetric()

            # Eval
            total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

            logger.info("***** Running evaluation *****")
            logger.info(f"  NOTE: if add_special_tokens = False")
            logger.info(f"  Num examples = {len(eval_dataset)}")
            logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
            logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}")
            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

            model.eval()

            # for error analysis
            all_predictions = []
            all_targets = []
            # all_indices = []

            for batch in eval_dataloader:
                    
                model_inputs = {
                    k: batch[k]
                    for k in ["input_ids", "attention_mask", "aug_input_ids", "aug_attention_mask"] # do not pass in labels
                    # for k in ["input_ids", "attention_mask", "labels", "aug_input_ids", "aug_attention_mask"] # omit aug_exist_idx
                }
                if batch['aug_input_ids'].shape[1] > 1 and 0 in batch["aug_exist_idx"]:
                    model_inputs["aug_exist_idx"] = batch["aug_exist_idx"]

                with torch.no_grad():
                    #TODO: greedy search:
                    outputs = model.generate(**model_inputs, num_beams = 1, do_sample = False)
                    #TODO: try out beam
                    # outputs = model.generate(**model_inputs, num_beams = 2)
                
                ####
                gts = batch['answer_input_ids']
                
                outputs = accelerator.pad_across_processes(outputs, dim=1)
                gts = accelerator.pad_across_processes(gts, dim=1)
                gts = accelerator.pad_across_processes(gts, dim=2)

                gathered_outputs = accelerator.gather(outputs)
                gathered_gts = accelerator.gather(gts)

                predictions_text_batch = tokenizer.batch_decode(gathered_outputs,skip_special_tokens=True)
                references_text_batch = [tokenizer.batch_decode(cands,skip_special_tokens=True) for cands in gathered_gts]

                # logger.info(predictions_text_batch)
                # logger.info(references_text_batch)

                metric.add_batch(
                    predictions=predictions_text_batch,
                    references=references_text_batch,
                )

                ## DEBUG:
                # logger.info(tokenizer.batch_decode(batch["input_ids"],skip_special_tokens=True))
                # logger.info(tokenizer.batch_decode(batch["aug_input_ids"][0],skip_special_tokens=True))
                # logger.info(tokenizer.batch_decode(outputs,skip_special_tokens=True))
                # logger.info(tokenizer.batch_decode(gts,skip_special_tokens=True))
                # logger.info('------------------------------------------------------------------')
                # if accelerator.is_main_process:
                #     all_predictions.append(
                #         {
                #             "input":tokenizer.batch_decode(batch["input_ids"],skip_special_tokens=True),
                #             "augmentation":tokenizer.batch_decode(batch["aug_input_ids"][0],skip_special_tokens=True),
                #             "prediction":tokenizer.batch_decode(outputs,skip_special_tokens=True),
                #             "ground_truth":tokenizer.batch_decode(gts,skip_special_tokens=True)
                #         }
                #     )
                #     with open('/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/eval/tmp/vis_NQ.json', 'w') as out:
                #         json.dump(all_predictions, out, indent=4)
                #     if len(all_predictions) == 100:
                #         quit()

                ## for error analysis
                # all_predictions += [int(item) for item in accelerator.gather(outputs).detach().cpu().numpy()]
                # all_targets += [int(item) for item in accelerator.gather(gts)]
                progress_bar.update(1)

            
            eval_metric = metric.compute()
            # eval_metric = metric.compute(ignore_case=True, ignore_punctuation=True)
            logger.info(eval_metric)
            
            accelerator.print(f"Result: {os.path.basename(dataset_path)} {eval_metric}")
            results = {
                "eval_file": dataset_path,
                "evaluation": eval_metric
            }
            if accelerator.is_main_process:
                step_output_dir = os.path.join(args.output_dir, os.path.basename(ckpt_path)) 
                os.makedirs(step_output_dir, exist_ok=True)
                output_path = os.path.join(step_output_dir, f"{os.path.basename(os.path.dirname(dataset_path))}__{os.path.basename(dataset_path)}")
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

    # unit_test()
