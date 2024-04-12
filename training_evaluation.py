import ast
import copy
import datetime
import pickle

import numpy as np
import torch
from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel


class WiCEvaluator(SentenceEvaluator):
    def __init__(self, dev_datasets, tokenizer, device, prefix, results_path, tmp_model_path_run,
                 save_every_epoch=False):
        """
        :param dev_datasets : dict of the form {name_of_the_dataset : dataframe}
        :param tokenizer  : Transformers tokenizer
        :param device : device on which the calculations should be made
        :param prefix : spicific name for the run
        :param results_path : path where to store the predictions and the scores
        :param tmp_model_path_run : path where to store the model for the evaluation
        :param save_every_epoch : whether to save the model after every epoch or not
        """
        self.tokenizer = tokenizer
        self.dev_datasets = dev_datasets
        self.device = device
        self.prefix = prefix
        self.results_path = results_path
        self.tmp_model_path_run = tmp_model_path_run
        self.save_every_epoch = save_every_epoch

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        print(f'Evaluation after epoch {epoch + 1}')
        now = datetime.datetime.now()
        print(now)
        tmp_model_path = f'{self.tmp_model_path_run}_tmp_eval_model'
        copy.deepcopy(model).save(tmp_model_path)
        if self.save_every_epoch:
            copy.deepcopy(model).save(tmp_model_path + f'_epoch{epoch + 1}')
        tmp_model = AutoModel.from_pretrained(tmp_model_path).to(self.device)

        overall_results = dict()
        for df_name, df in self.dev_datasets.items():
            print(f'Computing embeddings... ({df_name} ({df.shape[0]} elements))')
            all_embeddings = []
            for n, row in df.iterrows():
                word_borders1 = ast.literal_eval(row['Border1'])
                word_borders2 = ast.literal_eval(row['Border2'])
                tokenized1 = self.tokenizer(row['CutExample1'], return_offsets_mapping=True)
                tokenized2 = self.tokenizer(row['CutExample2'], return_offsets_mapping=True)
                offset1 = tokenized1['offset_mapping'][1:-1]
                offset2 = tokenized2['offset_mapping'][1:-1]
                token_idxs1 = [i for i, token_span in enumerate(offset1)
                               if word_borders1[0] <= token_span[0] <= word_borders1[1]]
                if len(token_idxs1) == 1:
                    token_pos1 = (token_idxs1[0] + 1, token_idxs1[0] + 2)
                else:
                    token_pos1 = (token_idxs1[0] + 1, token_idxs1[-1] + 2)
                token_idxs2 = [i for i, token_span in enumerate(offset2)
                               if word_borders2[0] <= token_span[0] <= word_borders2[1]]
                if len(token_idxs2) == 1:
                    token_pos2 = (token_idxs2[0] + 1, token_idxs2[0] + 2)
                else:
                    token_pos2 = (token_idxs2[0] + 1, token_idxs2[-1] + 2)

                tokenized1 = self.tokenizer(row['CutExample1'], return_offsets_mapping=False, return_tensors='pt').to(
                    self.device)
                tokenized2 = self.tokenizer(row['CutExample2'], return_offsets_mapping=False, return_tensors='pt').to(
                    self.device)
                out1 = tmp_model(**tokenized1)
                out2 = tmp_model(**tokenized2)
                emb1 = torch.mean(out1[0][0][token_pos1[0]:token_pos1[1]], dim=0).cpu().detach().numpy()
                emb2 = torch.mean(out2[0][0][token_pos2[0]:token_pos2[1]], dim=0).cpu().detach().numpy()
                all_embeddings.append((emb1, emb2,))

            print('Embeddings done...')

            true = [1 if x == True else 0 for x in df['Label'].tolist()]
            X1 = np.array([x[0] for x in all_embeddings])
            X2 = np.array([x[1] for x in all_embeddings])
            best_threshold, best_acc, pred = self._compute_accuracy(X1, X2, true)
            overall_results[df_name] = dict()
            overall_results[df_name]['no reduction'] = {'threshold': best_threshold, 'acc': best_acc, 'pred': pred}

            for n_comp in list(range(100, 101, 100)):
                # for n_comp in list(range(100, 701, 100)):
                for whiten_val in [True, False]:
                    X1 = np.array([x[0] for x in all_embeddings])
                    X2 = np.array([x[1] for x in all_embeddings])

                    pca = PCA(n_components=n_comp, whiten=whiten_val, random_state=42)
                    pca.fit(np.concatenate([X1, X2]))
                    reduced_X1 = pca.transform(X1)
                    reduced_X2 = pca.transform(X2)
                    reduced_best_threshold, reduced_best_acc, reduced_pred = self._compute_accuracy(reduced_X1,
                                                                                                    reduced_X2, true)
                    overall_results[df_name][f'{n_comp}_{whiten_val}'] = {'threshold': reduced_best_threshold,
                                                                          'acc': reduced_best_acc, 'pred': reduced_pred}

            print(f'Best Threshold {best_threshold}, accuracy: {round(best_acc * 100, 3)}% ({df_name} {self.prefix})')
            with open(self.results_path, 'a') as f:
                f.write(
                    f'{df_name} {self.prefix}_{epoch + 1} Best Threshold {best_threshold}, accuracy: {round(best_acc * 100, 3)}%\n')
            with open(self.results_path[:-4] + f'_{self.prefix}_epoch{epoch + 1}.pkl', 'wb') as results_file:
                pickle.dump(overall_results, results_file)
        now = datetime.datetime.now()
        print(now)
        return best_acc

    def _compute_accuracy(self, X1, X2, true):
        best_threshold, best_acc, best_pred = 0, 0, []
        for threshold in [x / 100 for x in range(0, 101, 2)][::-1]:
            pred = []
            for emb1, emb2 in zip(X1, X2):
                out = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))
                if out[0][0] > threshold:
                    pred.append(1)
                else:
                    pred.append(0)
            acc = accuracy_score(true, pred)
            if acc > best_acc:
                best_threshold, best_acc = threshold, acc
                best_pred = pred
        return best_threshold, best_acc, best_pred
