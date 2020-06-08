import torch
import torch.nn as nn
import torch.nn.functional as F


class DietNet(nn.Module):
    def __init__(self, feature_matrix, num_classes, device='cpu', number_of_genes=20530,
                 number_of_gene_features=173, embedding_size=500):
        """
        Class for the DietNetworks
        :param feature_matrix: Precomputed matrix of gene features of size (n_genes, n_features)
        :param num_classes: Number of classes in the prediction class
        """
        super(DietNet, self).__init__()
        self.embedding_size = embedding_size
        self.number_of_gene_features = number_of_gene_features
        self.number_of_genes = number_of_genes
        self.feature_matrix = torch.tensor(feature_matrix, requires_grad=False).float().to(device)

        self.predictor = nn.Linear(self.embedding_size, num_classes)

        # Define the parameters of the first auxiliary network (which predicts the parameters of the encoder)
        self.aux1_layer1 = nn.Linear(self.number_of_gene_features, self.embedding_size)
        self.aux1_layer2 = nn.Linear(self.embedding_size, self.embedding_size)

        # Define the parameters of the second auxiliary network (which predicts the parameters of the decoder)
        self.aux2_layer1 = nn.Linear(self.number_of_gene_features, self.embedding_size)
        self.aux2_layer2 = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, x):
        # Compute the weights of the encoder W_e using the auxiliary network 1
        W_e = self.aux1_layer2(F.relu(self.aux1_layer1(self.feature_matrix.T).T).T)
        latent = F.relu(torch.matmul(x, W_e))

        # Compute the weights of the decoder W_d using the auxiliary network 2
        W_d = self.aux2_layer2(F.relu(self.aux2_layer1(self.feature_matrix.T)))
        x_hat = torch.matmul(latent, W_d.T)

        y_hat = self.predictor(latent)

        return x_hat, y_hat