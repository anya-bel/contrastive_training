import random

import torch


class SenseContrastLoss(torch.nn.Module):
    def __init__(self, model, temperature=1, target_vector=False, self_augmentation=False, sim_score='cosine',
                 mean_last_layers=False):
        """

        :param model: SentenceTransformers model
        :param temperature: temperature parameter for the loss function
        :param target_vector: whether to use only the verb's vector to compute the similarity or the mean sentence embedding
        :param self_augmentation: whether to use self-augmentation to create positive examples for the senses which have only one example
        :param sim_score: which similarity score to use (cosine, euclidean or dot)
        :param mean_last_layers: whether to use the mean of last four layers for the computation or only the last layer
        """
        super(SenseContrastLoss, self).__init__()
        self.model = model
        self.temperature = temperature
        self.target_vector = target_vector
        self.self_augmentation = self_augmentation
        self.mean_last_layers = mean_last_layers
        if sim_score == 'cosine':
            self.score = self._cos_sim
        elif sim_score == 'euclidean':
            self.score = self._euc_sim
        elif sim_score == 'dot':
            self.score = self._dot_score
        else:
            raise ValueError(f'Incorrect score name {sim_score}')

    def forward(self, tokenized_sentences, labels):
        if self.self_augmentation:
            for k, v in tokenized_sentences[0].items():
                tokenized_sentences[0][k] = torch.cat([v, v], dim=0)
            labels = torch.cat([labels, labels], dim=0)
        if self.target_vector:
            features = self.model(tokenized_sentences[0])
            vectors = self._compute_target_vector(features, self.mean_last_layers)['target']
        else:
            vectors = self.model(tokenized_sentences[0])['sentence_embedding']
        sim_matrix = torch.div(self.score(vectors, vectors), self.temperature)  # .fill_diagonal_(0)
        # remove diagonal because of exp(0) in logsumexp
        new_sim_matrix = self._remove_diagonal(sim_matrix)
        denominator_all_j = torch.logsumexp(new_sim_matrix, dim=1, keepdim=False)
        loss = 0
        loss_change = False
        for j in range(len(vectors)):
            sense_j = labels[j]
            samesense_mask = labels == sense_j
            # remove the example j itself
            samesense_mask[j] = False
            samesense_vectors = vectors[samesense_mask]
            samesense_size = samesense_vectors.size(0)
            if samesense_size == 0:
                continue
            # denominator = torch.div(denominator_all_j[j], samesense_size)
            denominator = denominator_all_j[j]
            sim_matrix_num = torch.div(self.score(vectors[j], samesense_vectors), self.temperature)
            numerator = torch.div(-sim_matrix_num.sum(), samesense_size)
            loss += torch.add(denominator, numerator)
            loss_change = True
        if not loss_change:
            loss += torch.tensor(0, dtype=torch.float64, requires_grad=True)
        return loss

    @staticmethod
    def _compute_target_vector(features, mean_last_layers):
        borders = features['borders']
        offset_mapping = features['offset_mapping']
        if mean_last_layers:
            token_embs = torch.mean(torch.stack(features['all_layer_embeddings'][9:]), dim=0)
        else:
            token_embs = features['token_embeddings']
        attention_mask = features['attention_mask']

        output_vectors = []
        for tok_emb, offset, mask, border in zip(token_embs, offset_mapping, attention_mask, borders):
            text_offset = offset[mask == True][:-1]
            token_idxs = [i for i, token_span in enumerate(text_offset)
                          if border[0] <= token_span[0] <= border[1]]
            try:
                layer_pos_start = token_idxs[0]
                layer_pos_end = token_idxs[-1]
            except:
                print(offset, border, token_idxs)
            if layer_pos_start == 0:
                # 1 to get rid of the first element which is CLS
                layer_pos_start = 1
            mean_target = tok_emb[layer_pos_start:layer_pos_end + 1].mean(0).unsqueeze(0)
            output_vectors.append(mean_target)

        output_vector = torch.cat(output_vectors, 0)
        features.update({'target': output_vector})
        return features

    @staticmethod
    def _cos_sim(a, b):
        """
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py

        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    @staticmethod
    def _euc_sim(a, b):
        """
        Computes the euclidean similarity euc_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = euc_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        return torch.div(1, torch.cdist(a, b, p=2) + 1)

    @staticmethod
    def _dot_score(a, b):
        """
        Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        return torch.mm(a, b.transpose(0, 1))

    @staticmethod
    def _remove_diagonal(similarity_matrix):
        """
        Removes diagonal from the matrix
        :return: matrix without one element in the second dimension
        """
        mask = torch.zeros_like(similarity_matrix, dtype=bool).fill_(True).fill_diagonal_(False)
        row_size = similarity_matrix.size(0)
        new_similarity_matrix = torch.zeros((row_size, row_size - 1))
        for num_row, row in enumerate(similarity_matrix):
            new_similarity_matrix[num_row] = row[mask[num_row]]
        return new_similarity_matrix


class SenseDataLoader:

    def __init__(self, examples_with_labels, batch_size, self_augmentation=False):
        """
        A special data loader to be used with SenseContrastLoss.
        The data loader insures that one batch contains examples of a single lemma
        :param self_augmentation: whether to create a copy of each example or not
        """
        self.data_pointer = 0
        self.train_examples = examples_with_labels
        self.batch_size = batch_size
        self.self_augmentation = self_augmentation
        random.shuffle(self.train_examples)

    def __iter__(self):
        batch = self.train_examples[self.data_pointer]
        if self.self_augmentation:
            if len(batch) > self.batch_size / 2:
                batch = random.sample(batch, int(self.batch_size / 2))
            batch = copy.deepcopy(batch) + copy.deepcopy(batch)

        if len(batch) > self.batch_size:
            batch = random.sample(batch, self.batch_size)
        self.data_pointer += 1
        if self.data_pointer >= len(self.train_examples):
            self.data_pointer = 0
            random.shuffle(self.train_examples)

        yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def __len__(self):
        return len(self.train_examples)
