import argparse

import utils
from uda_trainer import UDATrainer
from cifar_trainer import CIFARTrainer


def main(config, args):

    utils.seed_everything(args.seed, debug=args.debug)

    # Setup directories
    config_name = config["exp_name"]
    output_dir_path = utils.create_folders(config_name)

    if "trainer_name" not in config:
        raise RuntimeError("\"trainer_name\" is not specified in the config file!")
    elif config["trainer_name"] == "UDATrainer":
        trainer = UDATrainer(config, output_dir_path, args.pretrained)
    elif config["trainer_name"] == "CIFARTrainer":
        trainer = CIFARTrainer(config, output_dir_path, args.pretrained)
    else:
        raise ValueError(f"Unsupported trainer name: {config['trainer_name']}!")

    if args.checkpoint_path is not None:
        trainer.load(args.checkpoint_path)
    trainer.run()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch UDA")
    args.add_argument("config",
                      default="config/uda_trainer.yaml",
                      type=str,
                      help="Path to the config file")
    args.add_argument("--checkpoint_path",
                      default=None,
                      type=str,
                      help="Path to weights file (default: None)")
    args.add_argument("--pretrained",
                      action="store_true",
                      help="Finetune pretrained weights")
    args.add_argument("--seed",
                      default=1234,
                      type=int,
                      help="Set random seed")
    args.add_argument("--debug",
                      action="store_true",
                      help="Run in debug mode")
    args = args.parse_args()

    config = utils.read_yaml(args.config)
    main(config, args)
