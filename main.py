import argparse
from data.data_loader import *
from train import *
from test import *
from parse_config import ConfigParser
import pickle

logger = logging.getLogger(__name__)
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--train_path', help='train path')
    parser.add_argument("--roberta_hidden_dim", default=512, type=int, help="Total batch size for training.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--seed", default=21, type=int, help="Random state")
    parser.add_argument("--kernel", default=21, type=int, help="Number of kernels")
    parser.add_argument("--sigma", default=1e-1, type=float, help="Sigma value used")
    parser.add_argument("--kfold_index", default=-1, type=int, help="Run this for K-fold cross validation")
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--postpretrain')

    args = parser.parse_args()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # config file
    config = configparser.ConfigParser(allow_no_value=True)
    config = configparser.ConfigParser()
    args = parser.parse_args()
    config.read(osp.join("config", args.config_file))
    args = set_args_from_config(args, config)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    timestr = datetime.datetime.now().strftime("%m-%d_%H")
    # set experiment name
    experiment_name = f"{'R' if args.dataset == 'Rumor-S' else 'G'}_{timestr}_{args.mode}_batch{args.train_batch_size}_{args.num_train_epochs}_lr{args.learning_rate}_K{args.kernel}_Sig{args.sigma}{'_KFold' + str(args.kfold_index) if args.kfold_index >= 0 else ''}_{args.sample_suffix}"

    if not osp.exists(osp.join("..", "logs", experiment_name)):
        os.mkdir(osp.join("..", "logs", experiment_name))

    handlers = [logging.FileHandler(osp.join("..", "logs", experiment_name, f"log_{experiment_name}.txt")), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG, datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logger.info(f'{args.dataset} Start training!')
    logger.info(f'Using batch size {args.train_batch_size} | accumulation {args.gradient_accumulation_steps}')

    label_map = {
        'real': 0,
        'fake': 1
    }
    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_pretrain, do_lower_case=False)
    trainset_reader, validset_reader = get_train_test_readers(label_map, tokenizer, args)

    # initial roberta model
    logger.info('Initializing roberta model')
    roberta_model = Roberta.from_pretrained(args.Roberta_pretrain)
    if args.postpretrain:
        model_dict = roberta_model.state_dict()
        pretrained_dict = torch.load(args.postpretrain)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)

    if args.cuda:
        # SD_model = 'Llama-2-7b'
        rl_model = SelectionPolicy()
        RV_model = layer_model()
        rl_model = rl_model.cuda()
        RV_model = RV_model.cuda()

    train_model(rl_model, RV_model, args, trainset_reader, validset_reader, writer=writer,
                experiment_name=experiment_name, llm = 'Llama-2-7b')
    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_pretrain, do_lower_case=False)
    logger.info("loading validation set")

    _, validset_reader = get_train_test_readers(label_map, tokenizer, args, test=True)

    logger.info('initializing estimator model')
    roberta_model = Roberta.from_pretrained(args.roberta_pretrain)
    roberta_model.eval()

    if args.cuda:
        rl_model = GeneratePolicy()
        RV_model = layer_model()
        rl_model = rl_model.cuda()
        RV_model = RV_model.cuda()

    eval_model(model, validset_reader, results_eval=results_eval, args=args)

