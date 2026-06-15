import argparse
import torch
import os
import swanlab
from allrec.utils import get_pipeline, parse_command_line_args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MMPGR', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--category', type=str, default='Toys_and_Games', help='Category name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    return parser.parse_known_args()

if __name__ == '__main__':
    args,unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    try:
        PipelineCls = get_pipeline(args.model)
        pipeline = PipelineCls(
            model_name=args.model,
            dataset_name=args.dataset,
            category=args.category,
            checkpoint_path=args.checkpoint,
            config_dict=command_line_configs
        )
        pipeline.run()
    except KeyboardInterrupt:
        if int(os.environ.get("RANK", 0)) == 0:
            print("\n[Main] User interrupted execution; cleaning up distributed state...")
    except Exception as e:
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"\n[Main] Exception: {e}")
        raise e
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            print(f"[Main] Rank {os.environ.get('RANK', 0)}: distributed process group destroyed.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if swanlab.get_run() is not None and int(os.environ.get("RANK", 0)) == 0:
            swanlab.finish()
            print("[Main] Rank 0: SwanLab experiment finished.")
