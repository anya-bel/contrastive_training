import argparse
import ast
import logging
import math
import os

import pandas as pd
import torch
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample
from sentence_transformers import models
from transformers import BertTokenizerFast

from sense_loss import SenseContrastLoss, SenseDataLoader
from training_evaluation import WiCEvaluator

# torch.cuda.set_device(0)
# device = 'cuda:0'
# print("GPU: ", torch.cuda.current_device())
device = 'cpu'

train_batch_size = 64
max_seq_length = 75

wikt_verb_dataset = pd.read_csv('datasets/wikt_train_set.csv').loc[:101]
wiktwic_df = pd.read_csv('datasets/wikt_wic_dev_set.csv').loc[:101]
wic_df = pd.read_csv('datasets/original_wic_dev_set.csv').loc[:101]

examples_labels = []
for verb, verb_df in wikt_verb_dataset.groupby('Verb'):
    if verb_df.shape[0] == 1:
        # removing single examples
        continue
    lab_list = verb_df['Definition']
    if len(set(lab_list)) == len(lab_list):
        # removing verbs with senses which have a single example
        continue
    ex_list = verb_df['CutExample']
    border_list = verb_df['Border']
    examples = [(InputExample(texts=[t], label=l), ast.literal_eval(b)) for t, l, b in
                zip(ex_list, lab_list, border_list)]

    examples_labels.append(examples)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def make_experiment(decay, lr, temperature, selfaug_val, n_runs, num_epochs, tmp_path, output_name):
    for run in range(n_runs):
        run += 1
        if not os.path.exists('model_results/'):
            os.mkdir('model_results/')
        output_file = 'model_results/' + output_name + '.txt'
        with open(output_file, 'w') as f:
            f.write('\n')

        print(
            f'Parameters:\nRun: {run}\nTemperature: {temperature}\nDecay: {decay}\nLR: {lr}\nSelf-Aug: {selfaug_val}\nEpochs: {num_epochs}')

        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=max_seq_length,
                                                  model_args={'output_hidden_states': True}).to(device)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode='mean').to(device)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(device)

        train_dataloader = SenseDataLoader(examples_labels, train_batch_size, self_augmentation=selfaug_val)

        dev_evaluator = WiCEvaluator({'wikiwic': wiktwic_df,
                                      'wic': wic_df},
                                     tokenizer,
                                     device,
                                     f'run{run}_decay{decay}_lr{lr}_selfaug{selfaug_val}_temp{temperature}',
                                     output_file,
                                     tmp_path+f'_run{run}',
                                     save_every_epoch=True)

        # Our training loss
        train_loss = SenseContrastLoss(model, target_vector=True, temperature=temperature)

        torch.autograd.set_detect_anomaly(True)

        # Configure the training
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        # Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=dev_evaluator,
                  epochs=num_epochs,
                  warmup_steps=warmup_steps,
                  output_path=tmp_path,
                  use_amp=False,  # Set to True, if your GPU supports FP16 operations
                  optimizer_params={'lr': lr},
                  weight_decay=decay
                  )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune on wiktionary')
    parser.add_argument("--decay", type=float, help='weight decay',
                        default=0.01)
    parser.add_argument("--lr", type=float, help='learning rate',
                        default=5e-6)
    parser.add_argument('--temperature', type=float, help='temperature to test',
                        default=0.5)
    parser.add_argument('--selfaug', type=str, help='if apply self-augmentation',
                        default=False)
    parser.add_argument('--n_runs', type=int, help='how many runs are needed',
                        default=5)
    parser.add_argument('--num_epochs', type=int, help='how many epochs are needed',
                        default=3)
    parser.add_argument('--tmp_path', type=str, help='path for the bert model',
                        default='sensebert')
    parser.add_argument('--output_name', type=str, help='name for the files with the results of model',
                        default='result')

    args = parser.parse_args()
    selfaug_val = True if args.selfaug == 'True' else False

    make_experiment(args.decay, args.lr, args.temperature, selfaug_val, args.n_runs, args.num_epochs, args.tmp_path,
                    args.output_name)
