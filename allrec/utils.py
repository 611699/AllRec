import importlib
import os
import re
import sys
import yaml
import html
import hashlib
import datetime
import requests
import tiktoken
from typing import Union, Optional
import logging
from logging import getLogger
from allrec.model import AbstractModel
from allrec.dataset import AbstractDataset
from accelerate.utils import set_seed

def init_seed(seed, reproducibility):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M")
    return cur

def get_command_line_args_str():
    return '-'.join(sys.argv[1:]).replace('/', '|').replace('--', '')

def get_file_name(config: dict, suffix: str = ''):
    stable_config = config_for_log(config)
    config_str = "".join([str(value) for _, value in sorted(stable_config.items(), key=lambda item: item[0])])
    md5 = hashlib.md5(config_str.encode(encoding="utf-8")).hexdigest()[:6]
    command_line_args = get_command_line_args_str()
    logfilename = "{}-{}-{}{}".format(
        config["run_id"], command_line_args, md5, suffix
    )
    return logfilename

def init_logger(config: dict):
    LOGROOT = config['log_dir']
    os.makedirs(LOGROOT, exist_ok=True)
    model_name = os.path.join(LOGROOT, config["model"])
    os.makedirs(model_name, exist_ok=True)
    dataset_name = os.path.join(model_name, config["dataset"])
    os.makedirs(dataset_name, exist_ok=True)
    category_name = os.path.join(dataset_name, config.get('category'))
    os.makedirs(category_name, exist_ok=True)
    logfilename = get_file_name(config, suffix='.log')
    if len(logfilename) > 255:
        logfilename = logfilename[:245] + logfilename[-10:]
    logfilepath = os.path.join(
        LOGROOT, config["model"],config["dataset"], config['category'], logfilename)
    if config['accelerator'].is_main_process:
        fh = logging.FileHandler(logfilepath)
        fh.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, handlers=[fh], force=True)
    else:
        logging.basicConfig(level=logging.INFO, handlers=[logging.NullHandler()], force=True)
    if not config['accelerator'].is_main_process:
        from datasets.utils.logging import disable_progress_bar
        disable_progress_bar()

def log(message, accelerator, logger, level='info'):
    if accelerator.is_main_process:
        if level == 'info':
            logger.info(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'debug':
            logger.debug(message)
        else:
            raise ValueError(f'Invalid log level: {level}')

def get_tokenizer(model_name: str):
    try:
        tokenizer_class = getattr(
            importlib.import_module(f'allrec.models.{model_name}.tokenizer'),
            f'{model_name}_Tokenizer'
        )
    except:
        raise ValueError(f'Tokenizer for model "{model_name}" not found.')
    return tokenizer_class

def get_model(model_name: Union[str, AbstractModel]) -> AbstractModel:
    if isinstance(model_name, AbstractModel):
        return model_name
    try:
        model_class = getattr(
            importlib.import_module(f'allrec.models.{model_name}.model'),
            f'{model_name}'
        )
    except:
        raise ValueError(f'Model "{model_name}" not found.')
    return model_class

def get_dataset(
    dataset_name: Union[str, AbstractDataset],
    model_name: Optional[Union[str, AbstractModel]] = None,
) -> AbstractDataset:
    if isinstance(dataset_name, AbstractDataset):
        return dataset_name
    if isinstance(model_name, str):
        try:
            dataset_class = getattr(
                importlib.import_module(f'allrec.models.{model_name}.dataset'),
                dataset_name,
            )
            return dataset_class
        except:
            pass
    try:
        dataset_class = getattr(
            importlib.import_module(f'allrec.datasets.{dataset_name}.dataset'),
            dataset_name
        )
    except:
        raise ValueError(
            f'Dataset "{dataset_name}" not found'
            + (
                f' for model "{model_name}"'
                if isinstance(model_name, str)
                else ''
            )
            + '.'
        )
    return dataset_class

def get_trainer(model_name: Union[str, AbstractModel]):
    from allrec.trainer import Trainer
    return Trainer

def maybe_log_custom_pipeline(pipeline_cls: type, config: dict) -> None:
    from allrec.pipeline import Pipeline as DefaultPipeline
    if pipeline_cls is DefaultPipeline:
        return
    accelerator = config.get('accelerator')
    if accelerator is None or not accelerator.is_main_process:
        return
    logger = getLogger()
    log(
        f"使用模型专用 Pipeline: {pipeline_cls.__name__} (模块 {pipeline_cls.__module__})",
        accelerator,
        logger,
        level='info',
    )

def get_pipeline(model_name: Union[str, AbstractModel]):
    from allrec.pipeline import Pipeline
    return Pipeline

def get_total_steps(config, train_dataloader):
    if config['steps'] is not None:
        return config['steps']
    else:
        return len(train_dataloader) * config['epochs']

def convert_config_dict(config: dict) -> dict:
    for key in config:
        v = config[key]
        if not isinstance(v, str):
            continue
        try:
            new_v = eval(v)
            if new_v is not None and not isinstance(
                new_v, (str, int, float, bool, list, dict, tuple)
            ):
                new_v = v
        except (NameError, SyntaxError, TypeError):
            if isinstance(v, str) and v.lower() in ['true', 'false']:
                new_v = (v.lower() == 'true')
            else:
                new_v = v
        config[key] = new_v
    return config

def get_config(
    model_name: Union[str, AbstractModel],
    dataset_name: Union[str, AbstractDataset],
    config_dict: Optional[dict]
) -> dict:
    final_config = {}
    current_path = os.path.dirname(os.path.realpath(__file__))
    config_file_list = [os.path.join(current_path, 'common.yaml')]
    config_file_list.append(
        os.path.join(current_path, f'datasets/{dataset_name}/config.yaml')
    )
    config_file_list.append(
        os.path.join(current_path, f'models/{model_name}/config.yaml'))
    for file in config_file_list:
        cur_config = yaml.safe_load(open(file, 'r'))
        if cur_config is not None:
            final_config.update(cur_config)
    if config_dict:
        final_config.update(config_dict)
    final_config['run_local_time'] = get_local_time()
    final_config = convert_config_dict(final_config)
    return final_config

def parse_command_line_args(unparsed: list[str]) -> dict:
    args = {}
    for text_arg in unparsed:
        if '=' not in text_arg:
            raise ValueError(
                f"Invalid command line argument: {text_arg}, "
                "please add '=' to separate key and value."
            )
        key, value = text_arg.split('=')
        if key.startswith('--'):
            key = key[len('--'):]
        elif key.startswith('-'):
            key = key[len('-'):]
        else:
            raise ValueError(f"Invalid argument format: {text_arg}. Expected '--key=value' or '-key=value'.")
        try:
            value = eval(value)
        except:
            pass
        args[key] = value
    return args

def download_file(url: str, path: str) -> None:
    logger = getLogger()
    response = requests.get(url,verify=False)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded {os.path.basename(path)}")
    else:
        logger.error(f"Failed to download {os.path.basename(path)}")

def list_to_str(l: Union[list, str], remove_blank=False) -> str:
    ret = ''
    if isinstance(l, list):
        ret = ', '.join(map(str, l))
    else:
        ret = l
    if remove_blank:
        ret = ret.replace(' ', '')
    return ret

def clean_text(raw_text: str) -> str:
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[^\x00-\x7F]', ' ', text)
    return text

def init_device():
    import torch
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_ddp = world_size > 1
    if torch.cuda.is_available():
        if use_ddp:
            torch.cuda.set_device(local_rank)
            return torch.device('cuda', local_rank), use_ddp
        return torch.device('cuda'), use_ddp
    else:
        return torch.device('cpu'), use_ddp

def config_for_log(config: dict) -> dict:
    config = config.copy()
    config.pop('device', None)
    config.pop('accelerator', None)
    for k, v in config.items():
        if isinstance(v, list):
            config[k] = str(v)
    return config

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
