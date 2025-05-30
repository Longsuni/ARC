from data_utils.utils import test_model
from models.modules import *
import numpy as np
import torch

class ARC():
    def __init__(self, args):
        self.beta_scheduler = ExponentialScheduler(start_value=args.beta_start_value, end_value=args.beta_end_value,
                                                   n_iterations=args.beta_n_iterations,
                                                   start_iteration=args.beta_start_iteration)
        self.hidden_size = args.hidden_dim
        self.POI_dim = args.POI_dim
        self.landUse_dim = args.landUse_dim
        self.region_num = args.region_num
        self.z_dim = args.z_dim
        self.encoder_z1 = Encoder(self.z_dim, self.POI_dim, self.hidden_size)
        self.encoder_z2 = Encoder(self.z_dim, self.landUse_dim, self.hidden_size)
        self.encoder_z3 = Encoder(self.z_dim, self.region_num, self.hidden_size)
        self.autoencoder_a = Decoder(self.z_dim, self.POI_dim, self.hidden_size)
        self.autoencoder_b = Decoder(self.z_dim, self.landUse_dim, self.hidden_size)
        self.autoencoder_c = Decoder(self.z_dim, self.region_num, self.hidden_size)

        self.mask_token_POI = nn.Parameter(torch.randn(self.POI_dim))
        self.mask_token_LandUse = nn.Parameter(torch.randn(self.landUse_dim))
        self.mask_token_region = nn.Parameter(torch.randn(self.region_num))
        self.mask_ratio = args.mask_ratio
        self.random_mask_ratio = args.random_mask_ratio
        self.best_r2 = 0
        self.iterations = 0
        self.best_emb = None
        device = torch.device('cuda:0')
        self.criterion = Loss(args.region_num, args.temperature, device).to(device)
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma



    def to_device(self, device):

        self.encoder_z1.to(device)
        self.encoder_z2.to(device)
        self.encoder_z3.to(device)
        self.autoencoder_a.to(device)
        self.autoencoder_b.to(device)
        self.autoencoder_c.to(device)

    def mask_input(self, x, random_mask_ratio=0.0):

        B, D = x.shape
        device = x.device

        random_mask = torch.rand(B, D, device=device) < random_mask_ratio

        x_masked = x.clone()

        if D == self.POI_dim:
            mask_token = self.mask_token_POI
        elif D == self.landUse_dim:
            mask_token = self.mask_token_LandUse
        elif D == self.region_num:
            mask_token = self.mask_token_region
        else:
            raise ValueError(f"error dim: {D}")

        mask_tokens = mask_token.unsqueeze(0).repeat(B, 1).to(device)

        num_rows_to_mask = max(1, int(self.mask_ratio * B))
        mask_row_indices = torch.randperm(B)[:num_rows_to_mask]

        row_mask = torch.ones(B, D, device=device)
        row_mask[mask_row_indices, :] = 0

        final_mask = row_mask * (~random_mask).float()

        x_masked[final_mask == 0] = mask_tokens[final_mask == 0]

        return x_masked, final_mask


    def mask_and_encode(self, x, encoder):
        x_masked, final_mask = self.mask_input(x,self.random_mask_ratio)
        z = encoder(x_masked)

        return z,final_mask

    def train_model(self, data, optimizer, task, city, sc, epochs):
        for epoch in range(epochs):
            poi_emb, landUse_emb, mob_emb = data
            poi_emb = poi_emb[0]
            landUse_emb = landUse_emb[0]
            mob_emb = mob_emb[0]

            z_1,mask_1 = self.mask_and_encode(poi_emb, self.encoder_z1)
            z_2,mask_2 = self.mask_and_encode(landUse_emb, self.encoder_z2)
            z_3,mask_3 = self.mask_and_encode(mob_emb, self.encoder_z3)

            de_z1 = self.autoencoder_a.decoder(z_1)
            de_z2 = self.autoencoder_b.decoder(z_2)
            de_z3 = self.autoencoder_c.decoder(z_3)

            raa_1 = reconstruction(poi_emb, de_z1, support='discrete')
            raa_2 = reconstruction(landUse_emb, de_z2, support='discrete')
            raa_3 = reconstruction(mob_emb, de_z3, support='discrete')
            mar_loss = raa_1 + raa_2 + raa_3


            sim_1_2, sim_1_3, sim_2_3 = calculate_pairwise_similarity(z_1, z_2, z_3, method='mmd')

            acl_1 = self.criterion.forward_feature_InfoNCE(z_1, z_2, batch_size=poi_emb.size(0))*sim_1_2
            acl_2 = self.criterion.forward_feature_InfoNCE(z_1, z_3, batch_size=poi_emb.size(0))*sim_1_3
            acl_3 = self.criterion.forward_feature_InfoNCE(z_2, z_3, batch_size=poi_emb.size(0))*sim_2_3

            cl_loss = acl_1 + acl_2 + acl_3

            ipr_1 = entropy(z_1, support='discrete')
            ipr_2 = entropy(z_2, support='discrete')
            ipr_3 = entropy(z_3, support='discrete')
            entropy_loss = ipr_1 + ipr_2 + ipr_3

            loss =  self.alpha * mar_loss + self.beta * entropy_loss + self.gamma * cl_loss

            latent_fusion = torch.cat([z_1, z_2, z_3], dim=1).cpu().detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sc.step(epoch)

            print("Epoch {}".format(epoch))
            self.test(latent_fusion, city, task)

            if epoch == epochs - 1:
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>best result:")
                self.test(self.best_emb, city, task)
                np.save("best_emb_{}_{}".format(city, task), self.best_emb)

    def test(self, latent_fusion, city, task):
        with ((torch.no_grad())):
            self.encoder_z1.eval(), self.encoder_z2.eval(), self.encoder_z3.eval()

            embs = latent_fusion

            _, _, r2 = test_model(city, task, embs)

            if self.best_r2 < r2:
                self.best_r2 = r2
                self.best_emb = embs

            self.encoder_z1.train()
            self.encoder_z2.train()
            self.encoder_z3.train()
            self.autoencoder_a.train()
            self.autoencoder_b.train()
            self.autoencoder_c.train()
