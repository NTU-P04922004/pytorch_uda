import argparse

import utils
from uda_trainer import UDATrainer


def main(config, checkpoint_path=None, pretrained=False):

    # Setup directories
    config_name = config["exp_name"]
    output_dir_path = utils.create_folders(config_name)

    trainer = UDATrainer(config, output_dir_path, pretrained)
    if checkpoint_path is not None:
        trainer.load(checkpoint_path)
    trainer.run()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch UDA")
    args.add_argument("config",
                      default="config.yaml",
                      type=str,
                      help="config file path")
    args.add_argument("--checkpoint_path",
                      default=None,
                      type=str,
                      help="path to weights file (default: None)")
    args.add_argument("--pretrained",
                      action="store_true",
                      help="Use pretrained weights")
    args = args.parse_args()

    config = utils.read_yaml(args.config)
    main(config, args.checkpoint_path, args.pretrained)
