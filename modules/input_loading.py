from utils import *
import torch.nn.functional as F


class MLUserLoading(torch.nn.Module): # sharing
    def __init__(self, embedding_dim):
        super(MLUserLoading, self).__init__()
        self.num_gender = config['n_gender']
        self.num_age = config['n_age']
        self.num_occupation = config['n_occupation']
        self.embedding_dim = embedding_dim

        self.embedding_gender = torch.nn.Embedding(num_embeddings=self.num_gender, embedding_dim=self.embedding_dim)
        self.embedding_age = torch.nn.Embedding(num_embeddings=self.num_age,embedding_dim=self.embedding_dim)
        self.embedding_occupation = torch.nn.Embedding(num_embeddings=self.num_occupation,embedding_dim=self.embedding_dim)

    def forward(self, x1):
        gender_idx, age_idx, occupation_idx = x1[:,0], x1[:,1], x1[:,2]
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        concat_emb = torch.cat((gender_emb, age_emb, occupation_emb), 1)
        return concat_emb


class MLItemLoading(torch.nn.Module): # sharing
    def __init__(self, embedding_dim):
        super(MLItemLoading, self).__init__()
        self.rate_dim = config['n_rate']
        self.genre_dim = config['n_genre']
        self.director_dim = config['n_director']
        self.year_dim = config['n_year']
        self.embedding_dim = embedding_dim

        self.emb_rate = torch.nn.Embedding(num_embeddings=self.rate_dim, embedding_dim=self.embedding_dim)
        self.emb_genre = torch.nn.Linear(in_features=self.genre_dim, out_features=self.embedding_dim, bias=False)
        self.emb_director = torch.nn.Linear(in_features=self.director_dim, out_features=self.embedding_dim, bias=False)
        self.emb_year = torch.nn.Embedding(num_embeddings=self.year_dim, embedding_dim=self.embedding_dim)

    def forward(self, x2):
        rate_idx, year_idx, genre_idx, director_idx = x2[:,0], x2[:,1], x2[:,2:27], x2[:,27:]
        rate_emb = self.emb_rate(rate_idx)
        year_emb = self.emb_year(year_idx)
        genre_emb = F.sigmoid(self.emb_genre(genre_idx.float()))
        director_emb = F.sigmoid(self.emb_director(director_idx.float()))
        concat_emb = torch.cat((rate_emb, year_emb, genre_emb, director_emb), 1)
        return concat_emb

class BKUserLoading(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(BKUserLoading, self).__init__()
        self.age_dim = config['n_age_bk']
        self.location_dim = config['n_location']
        self.embedding_dim = embedding_dim

        self.emb_age = torch.nn.Embedding(num_embeddings=self.age_dim, embedding_dim=self.embedding_dim)
        self.emb_location = torch.nn.Embedding(num_embeddings=self.location_dim, embedding_dim=self.embedding_dim)

    def forward(self, x1):
        age_idx, location_idx = x1[:,0], x1[:,1]
        age_emb = self.emb_age(age_idx)
        location_emb = self.emb_location(location_idx)
        concat_emb = torch.cat((age_emb, location_emb), 1)
        return concat_emb

class BKItemLoading(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(BKItemLoading, self).__init__()
        self.year_dim = config['n_year_bk']
        self.author_dim = config['n_author']
        self.publisher_dim = config['n_publisher']
        self.embedding_dim = embedding_dim

        self.emb_year = torch.nn.Embedding(num_embeddings=self.year_dim, embedding_dim=self.embedding_dim)
        self.emb_author = torch.nn.Embedding(num_embeddings=self.author_dim, embedding_dim=self.embedding_dim)
        self.emb_publisher = torch.nn.Embedding(num_embeddings=self.publisher_dim, embedding_dim=self.embedding_dim)

    def forward(self, x2):
        author_idx, year_idx, publisher_idx = x2[:,0], x2[:,1], x2[:,2]
        year_emb = self.emb_year(year_idx)
        author_emb = self.emb_author(author_idx)
        publisher_emb = self.emb_publisher(publisher_idx)
        concat_emb = torch.cat((year_emb, author_emb, publisher_emb), 1)
        return concat_emb

