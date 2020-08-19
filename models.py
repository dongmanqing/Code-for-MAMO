from utils import *
from torch.utils.data import DataLoader


class BASEModel(torch.nn.Module):
    def __init__(self, input1_module, input2_module, embedding1_module, embedding2_module, rec_module):
        super(BASEModel, self).__init__()

        self.input_user_loading = input1_module
        self.input_item_loading = input2_module
        self.user_embedding = embedding1_module
        self.item_embedding = embedding2_module
        self.rec_model = rec_module

    def forward(self, x1, x2):
        pu, pi = self.input_user_loading(x1), self.input_item_loading(x2)
        eu, ei = self.user_embedding(pu), self.item_embedding(pi)
        rec_value = self.rec_model(eu, ei)
        return rec_value

    def get_weights(self):
        u_emb_params = get_params(self.user_embedding.parameters())
        i_emb_params = get_params(self.item_embedding.parameters())
        rec_params = get_params(self.rec_model.parameters())
        return u_emb_params, i_emb_params, rec_params

    def get_zero_weights(self):
        zeros_like_u_emb_params = get_zeros_like_params(self.user_embedding.parameters())
        zeros_like_i_emb_params = get_zeros_like_params(self.item_embedding.parameters())
        zeros_like_rec_params = get_zeros_like_params(self.rec_model.parameters())
        return zeros_like_u_emb_params, zeros_like_i_emb_params, zeros_like_rec_params

    def init_weights(self, u_emb_para, i_emb_para, rec_para):
        init_params(self.user_embedding.parameters(), u_emb_para)
        init_params(self.item_embedding.parameters(), i_emb_para)
        init_params(self.rec_model.parameters(), rec_para)

    def get_grad(self):
        u_grad = get_grad(self.user_embedding.parameters())
        i_grad = get_grad(self.item_embedding.parameters())
        r_grad = get_grad(self.rec_model.parameters())
        return u_grad, i_grad, r_grad

    def init_u_mem_weights(self, u_emb_para, mu, tao, i_emb_para, rec_para):
        init_u_mem_params(self.user_embedding.parameters(), u_emb_para, mu, tao)
        init_params(self.item_embedding.parameters(), i_emb_para)
        init_params(self.rec_model.parameters(), rec_para)

    def init_ui_mem_weights(self, att_values, task_mem):
        # init the weights only for the mem layer
        u_mui = task_mem.read_head(att_values)
        init_ui_mem_params(self.rec_model.mem_layer.parameters(), u_mui)

    def get_ui_mem_weights(self):
        return get_params(self.rec_model.mem_layer.parameters())


class LOCALUpdate:
    def __init__(self, your_model, u_idx, dataset, sup_size, que_size, bt_size, n_loop, update_lr, top_k, device):
        self.s_x1, self.s_x2, self.s_y, self.s_y0, self.q_x1, self.q_x2, self.q_y, self.q_y0 = load_user_info(u_idx,
                                                                                                              dataset,
                                                                                                              sup_size,
                                                                                                              que_size,
                                                                                                              device)
        user_data = UserDataLoader(self.s_x1, self.s_x2, self.s_y, self.s_y0)
        self.user_data_loader = DataLoader(user_data, batch_size=bt_size)
        self.model = your_model

        self.update_lr = update_lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.update_lr)

        # self.loss_fn = torch.nn.NLLLoss()
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.n_loop = n_loop
        self.top_k = top_k

        self.device = device

        self.s_x1, self.s_x2, self.s_y = self.s_x1.to(self.device), self.s_x2.to(self.device), self.s_y.to(self.device)
        self.q_x1, self.q_x2, self.q_y = self.q_x1.to(self.device), self.q_x2.to(self.device), self.q_y.to(self.device)

    def train(self):
        for i in range(self.n_loop):
            # on support set
            for i_batch, (x1, x2, y, y0) in enumerate(self.user_data_loader):
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                pred_y = self.model(x1, x2)
                loss = self.loss_fn(pred_y, y)
                self.optimizer.zero_grad()
                loss.backward()  # local theta updating
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()

        q_pred_y = self.model(self.q_x1, self.q_x2)
        self.optimizer.zero_grad()
        loss = self.loss_fn(q_pred_y, self.q_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)

        u_grad, i_grad, r_grad = self.model.get_grad()
        return u_grad, i_grad, r_grad

    def test(self):
        for i in range(self.n_loop):
            # on support set
            for i_batch, (x1, x2, y, y0) in enumerate(self.user_data_loader):
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                pred_y = self.model(x1, x2)
                loss = self.loss_fn(pred_y, y)
                self.optimizer.zero_grad()
                loss.backward()  # local theta updating
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()

        # D.I.Y your calculation for the results
        q_pred_y = self.model(self.q_x1, self.q_x2)  # on query set


def maml_train(raw_phi_u, raw_phi_i, raw_phi_r, u_grad_list, i_grad_list, r_grad_list, global_lr):
    phi_u = update_parameters(raw_phi_u, u_grad_list, global_lr)
    phi_i = update_parameters(raw_phi_i, i_grad_list, global_lr)
    phi_r = update_parameters(raw_phi_r, r_grad_list, global_lr)
    return phi_u, phi_i, phi_r


def user_mem_init(u_id, dataset, device, feature_mem, loading_model, alpha):
    path = 'data_processed/' + dataset + '/raw/'
    u_x1_data = pickle.load(open('{}sample_{}_x1.p'.format(path, str(u_id)), 'rb'))
    u_x1 = to_torch([u_x1_data]).to(device)
    pu = loading_model(u_x1)
    personalized_bias_term, att_values = feature_mem.read_head(pu, alpha)
    del u_x1_data, u_x1, pu
    return personalized_bias_term, att_values
