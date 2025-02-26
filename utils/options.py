import argparse

def read_command_line():

    parser = argparse.ArgumentParser('Training and evaluation script', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters

    # Training 
    parser.add_argument('--batch_size', type=int, required=False, default=10, help='Batch size')
    parser.add_argument('--optimizer', type=str, required=False, default='Adam', choices=['Adam', 'SGD'], help='Adam or SGD')
    parser.add_argument('--learning_rate', type=float, required=False, default=0.0001, help='Learning rate')
    parser.add_argument('--scheduler', type= bool, required=False, default=False, help='True for using a learning rate scheduler')
    parser.add_argument('--num_epochs', type=int, required=False, default=10, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, required=False, default=1, help='Number of workers')


    # Others
    parser.add_argument('--dir', type=str, default='.',
                    help='Data directory')
    parser.add_argument('--dataset_path', default='VisualSudoku', type=str, help='The path to the training data.')
    parser.add_argument("--neptune_project", type=str, help="Neptune project directory")
    parser.add_argument("--neptune_api_token", type=str, help="Neptune api token")
    parser.add_argument("--split", type=int, required=False, default=10, help="Data split")


    args = parser.parse_args()
    return args
