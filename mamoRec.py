# Author: Manqing Dong, 2020

from modules.input_loading import *
from modules.info_embedding import *
from modules.rec_model import *
from modules.memories import *
from models import *
from configs import *


class MAMRec:
    def __init__(self, dataset='movielens'):

        self.dataset = dataset
        self.support_size = config_settings['support_size']
        self.query_size = config_settings['query_size']
        self.n_epoch = config_settings['n_epoch']
        self.n_inner_loop = config_settings['n_inner_loop']
        self.batch_size = config_settings['batch_size']
        self.n_layer = config_settings['n_layer']
        self.embedding_dim = config_settings['embedding_dim']
        self.rho = config_settings['rho']  # local learning rate
        self.lamda = config_settings['lamda']  # global learning rate
        self.tao = config_settings['tao']  # hyper-parameter for initializing personalized u weights
        self.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device(config_settings['cuda_option'] if self.USE_CUDA else "cpu")
        self.n_k = config_settings['n_k']
        self.alpha = config_settings['alpha']
        self.beta = config_settings['beta']
        self.gamma = config_settings['gamma']
        self.active_func = config_settings['active_func']
        self.rand = config_settings['rand']
        self.random_state = config_settings['random_state']
        self.split_ratio = config_settings['split_ratio']

        # load dataset
        self.train_users, self.test_users = train_test_user_list(dataset=dataset, rand=self.rand,
                                                                 random_state=self.random_state,
                                                                 train_test_split_ratio=self.split_ratio)

        if dataset == 'movielens':
            self.x1_loading, self.x2_loading = MLUserLoading(embedding_dim=self.embedding_dim).to(self.device), \
                                               MLItemLoading(embedding_dim=self.embedding_dim).to(self.device)
        else:
            self.x1_loading, self.x2_loading = BKUserLoading(embedding_dim=self.embedding_dim).to(self.device), \
                                               BKItemLoading(embedding_dim=self.embedding_dim).to(self.device)

        self.n_y = default_info[dataset]['n_y']

        # Embedding model
        self.UEmb = UserEmbedding(self.n_layer, default_info[dataset]['u_in_dim'] * self.embedding_dim,
                                  self.embedding_dim, activation=self.active_func).to(self.device)
        self.IEmb = ItemEmbedding(self.n_layer, default_info[dataset]['i_in_dim'] * self.embedding_dim,
                                  self.embedding_dim, activation=self.active_func).to(self.device)

        # rec model
        self.rec_model = RecMAM(self.embedding_dim, self.n_y, self.n_layer, activation=self.active_func).to(self.device)

        # whole model
        self.model = BASEModel(self.x1_loading, self.x2_loading, self.UEmb, self.IEmb, self.rec_model).to(self.device)

        self.phi_u, self.phi_i, self.phi_r = self.model.get_weights()

        self.FeatureMEM = FeatureMem(self.n_k, default_info[dataset]['u_in_dim'] * self.embedding_dim,
                                     self.model, device=self.device)
        self.TaskMEM = TaskMem(self.n_k, self.embedding_dim, device=self.device)

        self.train = self.train_with_meta_optimization
        self.test = self.test_with_meta_optimization

        self.train()

    def train_with_meta_optimization(self):
        for i in range(self.n_epoch):
            u_grad_sum, i_grad_sum, r_grad_sum = self.model.get_zero_weights()

            # On training dataset
            for u in self.train_users[:100]:
                # init local parameters: theta_u, theta_i, theta_r
                bias_term, att_values = user_mem_init(u, self.dataset, self.device, self.FeatureMEM, self.x1_loading,
                                                      self.alpha)
                self.model.init_u_mem_weights(self.phi_u, bias_term, self.tao, self.phi_i, self.phi_r)
                self.model.init_ui_mem_weights(att_values, self.TaskMEM)

                user_module = LOCALUpdate(self.model, u, self.dataset, self.support_size, self.query_size, self.batch_size,
                                          self.n_inner_loop, self.rho, top_k=3, device=self.device)
                u_grad, i_grad, r_grad = user_module.train()

                u_grad_sum, i_grad_sum, r_grad_sum = grads_sum(u_grad_sum, u_grad), grads_sum(i_grad_sum, i_grad), \
                                                     grads_sum(r_grad_sum, r_grad)

                self.FeatureMEM.write_head(u_grad, self.beta)
                u_mui = self.model.get_ui_mem_weights()
                self.TaskMEM.write_head(u_mui[0], self.gamma)

            self.phi_u, self.phi_i, self.phi_r = maml_train(self.phi_u, self.phi_i, self.phi_r,
                                                            u_grad_sum, i_grad_sum, r_grad_sum, self.lamda)

            self.test_with_meta_optimization()

    def test_with_meta_optimization(self):
        best_phi_u, best_phi_i, best_phi_r = self.model.get_weights()

        for u in self.test_users:
            bias_term, att_values = user_mem_init(u, self.dataset, self.device, self.FeatureMEM, self.x1_loading,
                                                  self.alpha)
            self.model.init_u_mem_weights(best_phi_u, bias_term, self.tao, best_phi_i, best_phi_r)
            self.model.init_ui_mem_weights(att_values, self.TaskMEM)

            self.model.init_weights(best_phi_u, best_phi_i, best_phi_r)
            user_module = LOCALUpdate(self.model, u, self.dataset, self.support_size, self.query_size, self.batch_size,
                                      self.n_inner_loop, self.rho, top_k=3, device=self.device)
            user_module.test()
