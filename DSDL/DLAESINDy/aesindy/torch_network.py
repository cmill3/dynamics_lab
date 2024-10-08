import torch
import torch.nn as nn
import sys

#syspath = 'SindyPendulum/'
#if syspath not in sys.path:
    #sys.path.append(syspath)
    
from .sindy_library import SINDyLibrary



class Encoder(nn.Module):
    def __init__(self, input_dim,latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,latent_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))   
        # x = torch.relu(self.fc4(x))
        return x 

class Decoder(nn.Module):
    def __init__(self, input_dim,latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim,64)
        self.fc2 = nn.Linear(64,128)
        self.fc3 = nn.Linear(128,input_dim)
        self.initialize_weights()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))   
        # x = torch.relu(self.fc4(x))
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class SindyAutoencoder(nn.Module):
    def __init__(self,model_hyperparameters, device):
        super(SindyAutoencoder, self).__init__()
        self.encoder = Encoder(model_hyperparameters['input_dim'],model_hyperparameters['latent_dim'])
        self.decoder = Decoder(model_hyperparameters['input_dim'],model_hyperparameters['latent_dim'])
        self.model_hyperparameters = model_hyperparameters
        self.device = device
        self.SINDyLibrary = SINDyLibrary(
            device=self.device,
            latent_dim=model_hyperparameters['latent_dim'],
            include_biases=True,
            include_states=True,
            include_sin=model_hyperparameters['include_sine'],
            include_cos=False,
            include_multiply_pairs=True,
            poly_order=model_hyperparameters['poly_order'],
            include_sqrt=False,
            include_inverse=False,
            include_sign_sqrt_of_diff=False)
        

        self.XI = nn.Parameter(torch.full((self.SINDyLibrary.number_candidate_functions,model_hyperparameters['latent_dim']),1,dtype = torch.float32,requires_grad=True,device=device))
        self.XI_coefficient_mask = torch.ones((self.SINDyLibrary.number_candidate_functions,model_hyperparameters['latent_dim']),dtype = torch.float32, device=device)
        self.model_hyperparameters = model_hyperparameters


    # def t_derivative(self,input_value, xdot, weights, biases, activation='sigmoid'):
    #     """
    #     Compute the first order time derivatives by propagating through the network.
    #     da[l]dt = xdot * da[l]dx = xdot * product(g'(w[l]a[l-1] + b[l])* w[l])
    #     Arguments:
    #         input - 2D tensorflow array, input to the network. Dimensions are number of time points
    #         by number of state variables.
    #         xdot - First order time derivatives of the input to the network. quello che conosciamo
    #         weights - List of tensorflow arrays containing the network weights
    #         biases - List of tensorflow arrays containing the network biases
    #         activation - String specifying which activation function to use. Options are
    #         'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
    #         or linear.

    #     Returns:
    #         dadt - Tensorflow array, first order time derivatives of the network output.
    #     """
    #     a   = input_value
    #     dadt = xdot #per le condizioni iniziali

    #     if activation == 'sigmoid':
    #         for i in range(len(weights) - 1):
    #             z = torch.matmul(a, weights[i].T) + biases[i]
    #             a = torch.sigmoid(z)
    #             gprime = a * (1-a)
    #             dadt = gprime * torch.matmul(dadt, weights[i].T)
    #         dadt = torch.matmul(dadt, weights[-1].T) #fuori dal ciclo bisogna ancora moltiplicare per i pesi dell ultimo livello outside the cycle you still need to multiply by the weights of the last level
            
    #     elif activation == 'relu':
    #         for i in range(len(weights) - 1):
    #             z = torch.matmul(a, weights[i].T) + biases[i]
    #             a = torch.relu(z)
    #             dadt = (z > 0).float() * torch.matmul(dadt, weights[i].T)    
    #         dadt = torch.matmul(dadt, weights[-1].T) #fuori dal ciclo bisogna ancora moltiplicare per i pesi dell ultimo livello outside the cycle you still need to multiply by the weights of the last level
    #     return dadt #nel caso che ci serve dadt sará l output dell encoder ossia le latent variables! in case we need dadt it will be the encoder output i.e. the latent variables!

    
    
    # def compute_quantities(self,x,xdot):
    
    #     z = self.encoder(x)
    #     xtilde = self.decoder(z)

    #     theta = self.SINDyLibrary.transform(z) 
    #     zdot_hat = torch.matmul(theta, self.XI_coefficient_mask * self.XI)
        
    #     encoder_parameters = list(self.encoder.parameters())
    #     encoder_weight_list = [w for w in encoder_parameters if len(w.shape) == 2]
    #     encoder_biases_list = [b for b in encoder_parameters if len(b.shape) == 1]
    #     zdot = self.t_derivative(x, xdot, encoder_weight_list, encoder_biases_list, activation=self.model_hyperparameters['activation'])                                               

    #     #print("propagazione sul decoder")
    #     decoder_parameters = list(self.decoder.parameters())
    #     decoder_weight_list = [w for w in decoder_parameters if len(w.shape) == 2]
    #     decoder_biases_list = [b for b in decoder_parameters if len(b.shape) == 1]
    #     xtildedot = self.t_derivative(z, zdot_hat, decoder_weight_list, decoder_biases_list, activation=self.model_hyperparameters['activation'])    
        
    #     return xtilde, xtildedot, z, zdot, zdot_hat

    def compute_quantities(self, x, dx_dt):
        # Compute z and its time derivative
        z = self.encoder(x)
        with torch.enable_grad():
            x.requires_grad_(True)
            z = self.encoder(x)
            dz_dx = torch.autograd.functional.jacobian(self.encoder, x)
            
            # Get the actual shapes
            batch_size, input_dim = x.shape
            latent_dim = z.shape[1]
            
            # Reshape dz_dx to [batch_size, latent_dim, input_dim]
            dz_dx = dz_dx.reshape(batch_size, latent_dim, batch_size, input_dim)
            dz_dx = dz_dx.diagonal(dim1=0, dim2=2).permute(2, 0, 1)
            
            # Reshape dx_dt to [batch_size, input_dim, 1]
            dx_dt = dx_dt.unsqueeze(-1)
            
            # Perform batch matrix multiplication
            dz_dt = torch.bmm(dz_dx, dx_dt)
            dz_dt = dz_dt.squeeze(-1)

        # Compute xh and its time derivative
        xh = self.decoder(z)
        with torch.enable_grad():
            z.requires_grad_(True)
            xh = self.decoder(z)
            dxh_dz = torch.autograd.functional.jacobian(self.decoder, z)
            
            # Reshape dxh_dz to [batch_size, output_dim, latent_dim]
            output_dim = xh.shape[1]
            dxh_dz = dxh_dz.reshape(batch_size, output_dim, batch_size, latent_dim)
            dxh_dz = dxh_dz.diagonal(dim1=0, dim2=2).permute(2, 0, 1)
        
        # Compute SINDy prediction
        theta = self.SINDyLibrary.transform(z)
        dz_dt_sindy = torch.matmul(theta, self.XI_coefficient_mask * self.XI)
        
        # Compute time derivative of xh using SINDy prediction
        dxh_dt = torch.bmm(dxh_dz, dz_dt_sindy.unsqueeze(-1)).squeeze(-1)

        return xh, dxh_dt, z, dz_dt, dz_dt_sindy

    
    def get_loss_function(self):
        if self.model_hyperparameters["loss_function"] == "Huber":
            return nn.HuberLoss()
        elif self.model_hyperparameters["loss_function"] == "MSE":
            return nn.MSELoss()


    def loss_function(self, x, dx_dt, xh, dxh_dt, z, dz_dt, dz_dt_sindy):
        loss_function = self.get_loss_function()
        loss = {}
        loss['recon_loss'] = loss_function(x, xh)
        loss['sindy_loss_x'] = loss_function(dx_dt, dxh_dt) if self.model_hyperparameters['loss_weight_sindy_x'] > 0.0 else torch.tensor(0.0)
        loss['sindy_loss_z'] = loss_function(dz_dt, dz_dt_sindy) if self.model_hyperparameters['loss_weight_sindy_z'] > 0.0 else torch.tensor(0.0)
        loss['sindy_regular_loss'] = torch.sum(torch.abs(self.XI))
        loss['total_loss'] = (
            self.model_hyperparameters['loss_weight_rec'] * loss['recon_loss'] +
            self.model_hyperparameters['loss_weight_sindy_x'] * loss['sindy_loss_x'] +
            self.model_hyperparameters['loss_weight_sindy_z'] * loss['sindy_loss_z'] +
            self.model_hyperparameters['loss_weight_sindy_regularization'] * loss['sindy_regular_loss']
        )
        return loss['total_loss'], loss
    
    def forward(self, x, xdot):
        return self.compute_quantities(x, xdot)
    
    
    