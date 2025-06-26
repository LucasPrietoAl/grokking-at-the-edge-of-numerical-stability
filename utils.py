import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataset import random_split
from datasets import (AlgorithmicDataset, 
                      SparseParityDataset, 
                      BinaryAlgorithmicDataset,
                      AlgorithmicDatasetTransformer)
from models import MLP, Transformer
from binary_operations import (product_mod,
                               add_mod,
                               subtract_mod)
from constants import  FLOAT_PRECISION_MAP



def one_hot_encode(number, size):
    one_hot = torch.zeros(size)
    one_hot[number] = 1
    return one_hot

def cross_entropy_float64(logits, labels, reduction="mean"):
    labels = labels.to(torch.int64)
    logprobs = torch.nn.functional.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1).to(torch.float64)
    loss = -torch.mean(prediction_logprobs) if reduction=="mean" else - prediction_logprobs
    return loss.to(torch.float32)


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )

def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, reduction="mean"):
    labels = labels.to(torch.int64)
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1).to(torch.float64)

    loss = -torch.mean(prediction_logprobs) if reduction=="mean" else - prediction_logprobs
    return loss


def cross_entropy_float32(logits, labels, reduction="mean"):
    labels = labels.to(torch.int64)
    logprobs = torch.nn.functional.log_softmax(logits.to(torch.float32), dim=-1)
    labels = labels.view(-1, 1)
    prediction_logprobs = torch.gather(logprobs, dim=-1, index=labels)
    prediction_logprobs = prediction_logprobs.squeeze(-1)

    if reduction == "mean":
        loss = -torch.mean(prediction_logprobs)
    elif reduction == "sum":
        loss = -torch.sum(prediction_logprobs)
    elif reduction == "none":
        loss = -prediction_logprobs
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


def cross_entropy_float16(logits, labels, reduction="mean"):
    labels = labels.to(torch.int64)
    logprobs = torch.nn.functional.log_softmax(logits.to(torch.float16), dim=-1)

    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1).to(torch.float16)
    loss = -torch.mean(prediction_logprobs) if reduction=="mean" else - prediction_logprobs
    return loss

def update_results(filename, experiment_key, logger_metrics):
    try:
        results = torch.load(filename)
    except:
        results = {}
        
    results[experiment_key] = logger_metrics
    torch.save(results, filename)

def evaluate(model, data_loader, loss_function=cross_entropy_float64):
    model.eval()
    loss = 0
    correct = 0
    device = next(model.parameters()).device
    float_precision = next(model.parameters()).dtype
    with torch.no_grad():
        for data, target, *_ in data_loader:
            label_argmax = len(target.shape)!=1
            output = model(data.to(device).to(float_precision)).to("cpu")
            if isinstance(model, Transformer):
                output = output[:,-1]
            loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            if label_argmax:
                target = target.argmax(dim=1)
            correct += pred.eq(target.to("cpu").view_as(pred)).sum().item()
    loss /= len(data_loader)
    accuracy = 100 * correct / len(data_loader.dataset)
    return loss, accuracy



def get_specified_args(parser, args):

    defaults = {action.dest: action.default
                for action in parser._actions
                if action.dest != 'help'}
    
    specified = {arg: getattr(args, arg)
                 for arg in vars(args)
                 if getattr(args, arg) != defaults.get(arg)
                 and arg!="device"}
    
    return specified

def split_dataset(dataset, train_fraction, batch_size):
    total_size = len(dataset)
    train_size = int(train_fraction * total_size)
    test_size = total_size - train_size
    print(f'Starting training. Train dataset size: {train_size}, Test size: {test_size}')
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def reduce_train_dataset(original_train_dataset, reduced_fraction, batch_size):
    original_indices = original_train_dataset.indices
    reduced_train_size = int(reduced_fraction * len(original_indices))
    reduced_indices = original_indices[:reduced_train_size]
    reduced_train_dataset = Subset(original_train_dataset, reduced_indices)
    
    reduced_train_loader = DataLoader(reduced_train_dataset, batch_size=batch_size, shuffle=True)
    return reduced_train_loader

BINARY_OPERATION_MAP =  {"add_mod": add_mod,
                         "product_mod": product_mod,
                         "subtract_mod": subtract_mod
                         }
def get_dataset(args):
    if args.dataset == "sparse_parity":
        print(args.num_parity_features, args.num_noise_features)
        dataset = SparseParityDataset(args.num_parity_features, args.num_noise_features, args.num_samples)
        
    elif args.dataset == "binary_alg":
        dataset = BinaryAlgorithmicDataset(BINARY_OPERATION_MAP[args.binary_operation], p=args.modulo, input_size=args.input_size, output_size=args.modulo)
    else: 
        if args.use_transformer:
            dataset = AlgorithmicDatasetTransformer(BINARY_OPERATION_MAP[args.binary_operation], p=args.modulo, input_size=args.input_size, output_size=args.modulo)
        else:
            dataset = AlgorithmicDataset(BINARY_OPERATION_MAP[args.binary_operation], p=args.modulo, input_size=args.input_size, output_size=args.modulo)
    
    train_dataset, test_dataset = split_dataset(dataset, args.train_fraction, args.batch_size)

    return train_dataset, test_dataset

def generate_random_one_hot(length):
    index = torch.randint(0, length, (1,)).item()
    one_hot_vector = torch.zeros(length)
    one_hot_vector[index] = 1
    return one_hot_vector

def get_model(args):
    device = args.device

    if args.dataset == "sparse_parity":
        model = MLP(input_size= args.num_parity_features + args.num_noise_features, output_size=2, 
                    hidden_sizes=args.hidden_sizes).to(device) 

    elif args.dataset == "binary_alg":
        model = MLP(input_size=(args.input_size - 1).bit_length()*2, output_size=args.modulo, 
                    hidden_sizes=args.hidden_sizes).to(device)

    elif args.dataset == "scalar_alg":
        model = MLP(input_size=2, output_size=args.modulo, hidden_sizes=args.hidden_sizes).to(device)
                    
    else:
        print("Using AlgorithmicDataset")
        if args.use_transformer:
            model = Transformer(d_model=128, num_heads=4, num_layers=1, vocab_size=113, seq_len=2)
        else:
            model = MLP(input_size=args.input_size*2, output_size=args.modulo, hidden_sizes=args.hidden_sizes
                    , bias=False).to(device).to(FLOAT_PRECISION_MAP[args.train_precision])
    return model
        
def get_optimizer(model, args):
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0, eps=args.adam_epsilon)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_epsilon, betas=(0.9, args.beta2))
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.2, weight_decay=0)
    else: 
        raise ValueError(f'Unsupported optimizer type: {args.optimizer}')
    return optimizer
    
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network with specified parameters.")

    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[200, 200],
                        help='List of hidden layer sizes. Default is [200, 200].')

    parser.add_argument('--num_epochs', type=int, default=1500,
                        help='Number of epochs. Default is 1500.')

    parser.add_argument('--train_fraction', type=float, default=0.3,
                        help='Fraction of data to be used for training. Default is 0.3.')

    parser.add_argument('--modulo', type=int, default=113,
                        help='Modulo value for modular arithmetic datasets. Default is 113.')

    parser.add_argument('--input_size', type=int, default=113,
                        help='Input size for the model. Default is 113.')

    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='Optimizer to use. Options: AdamW, Adam, SGD. Default is AdamW.')

    parser.add_argument('--loss_function', type=str, default='cross_entropy',
                        help='Loss function to use. Options: stablemax, cross_entropy. Default is cross_entropy.')

    parser.add_argument('--log_frequency', type=int, default=50,
                        help='Logging frequency (in epochs). Default is 50.')

    parser.add_argument('--regularization', type=str, default="None",
                        help='Regularization method. Options: None, l1, l2. Default is None.')
    
    parser.add_argument('--binary_operation', type=str, default="add_mod",
                        help='Binary operation for algorithmic tasks. Options: add_mod, product_mod, subtract_mod')

    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Default is None.')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size. Default is 128.')

    parser.add_argument('--full_batch', action='store_true', default=True,
                        help='Use full batch gradient descent. Default is True.')

    parser.add_argument('--dataset', type=str, default="add_mod",
                        help='Dataset to use. Options: rotated_mnist, add_mod. Default is add_mod.')

    parser.add_argument('--temperature_schedule', action='store_true', default=False,
                        help='Use a schedule for softmax temperature. Default is False.')

    parser.add_argument('--num_noise_features', type=int, default=50,
                        help='Number of noise features used for SparseParityDataset. Default is 50.')

    parser.add_argument('--num_parity_features', type=int, default=4,
                        help='Number of parity features used for SparseParityDataset. Default is 4.')

    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples for SparseParityDataset. Default is 1000.')

    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Alpha coefficient that multiplies the logits. Default is 1.0.')

    parser.add_argument('--lambda_l1', type=float, default=0.00001,
                        help='L1 regularization coefficient. Default is 0.00001.')

    parser.add_argument('--lambda_l2', type=float, default=0.00005,
                        help='L2 regularization coefficient. Default is 0.00005.')

    parser.add_argument('--softmax_precision', type=int, default=32,
                        help='Floating point precision for the loss calculation: 16, 32, or 64. Default is 32.')
    
    parser.add_argument('--train_precision', type=int, default=32,
                        help='Floating point precision for the model and data: 16, 32, or 64. Default is 32.')

    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 penalty) coefficient. Default is 0.')

    parser.add_argument('--use_lr_scheduler', action='store_true', default=False,
                        help='Use a learning rate scheduler. Default is False.')

    parser.add_argument('--orthogonal_gradients', action='store_true', default=False,
                        help='Use orthogonal gradients regularization. Default is False.')
    
    parser.add_argument('--use_transformer', action='store_true', default=False,
                        help='Use one layer transformer')
    
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Beta2 parameter for Adam and AdamW')
    parser.add_argument('--adam_epsilon', type=float, default=1e-25,
                        help='Epsilon value for Adam and AdamW')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed. Default is 42.')

    return parser, parser.parse_args()
