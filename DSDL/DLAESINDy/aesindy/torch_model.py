import torch
import torch.nn as nn
import torch.nn.functional as F

class Sindy(nn.Module):
    def __init__(self, library_dim, state_dim, poly_order, model='lorenz', initializer='constant', 
                 actual_coefs=None, rfe_threshold=None, include_sine=False, exact_features=False, 
                 fix_coefs=False, sindy_pert=0.0, ode_net=False, ode_net_widths=[1.5, 2.0]):
        super(Sindy, self).__init__()
        
        self.library_dim = library_dim
        self.state_dim = state_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.rfe_threshold = rfe_threshold
        self.exact_features = exact_features
        self.fix_coefs = fix_coefs
        self.sindy_pert = sindy_pert
        self.model = model
        self.ode_net = ode_net
        self.ode_net_widths = ode_net_widths
        
        # Initialize coefficients
        if isinstance(initializer, torch.Tensor):
            self.coefficients_mask = nn.Parameter(torch.abs(initializer) > 1e-10, requires_grad=False)
            self.coefficients = nn.Parameter(initializer)
        else:
            if initializer == 'true' and actual_coefs is not None:
                self.coefficients_mask = nn.Parameter(torch.abs(torch.tensor(actual_coefs)) > 1e-10, requires_grad=False)
                self.coefficients = nn.Parameter(torch.tensor(actual_coefs) + sindy_pert * (torch.rand_like(torch.tensor(actual_coefs)) - 0.5))
            else:
                self.coefficients_mask = nn.Parameter(torch.ones((library_dim, state_dim)), requires_grad=False)
                if initializer == 'constant':
                    self.coefficients = nn.Parameter(torch.zeros((library_dim, state_dim)))
                else:
                    self.coefficients = nn.Parameter(torch.randn((library_dim, state_dim)))
        
        if self.fix_coefs:
            self.coefficients.requires_grad = False
        
        if self.ode_net:
            self.net_model = self.make_theta_network(self.library_dim, self.ode_net_widths)
    
    def make_theta_network(self, output_dim, widths):
        layers = []
        in_features = self.state_dim
        for w in widths:
            layers.append(nn.Linear(in_features, int(w * in_features)))
            layers.append(nn.ELU())
            in_features = int(w * in_features)
        layers.append(nn.Linear(in_features, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, z):
        theta = self.theta(z)
        return torch.matmul(theta, self.coefficients)
    
    def theta(self, z):
        if self.ode_net:
            return self.net_model(z)
        else:
            return self.sindy_library(z, self.state_dim, self.poly_order, self.include_sine, self.exact_features, self.model)
    
    def sindy_library(self, z, latent_dim, poly_order, include_sine=False, exact_features=False, model='lorenz'):
        if exact_features:
            if model == 'lorenz':
                library = [
                    z[:, 0],
                    z[:, 1],
                    z[:, 2],
                    z[:, 0] * z[:, 1],
                    z[:, 0] * z[:, 2]
                ]
            elif model == 'predprey':
                library = [
                    z[:, 0],
                    z[:, 1],
                    z[:, 0] * z[:, 1]
                ]
            elif model == 'rossler':
                raise NotImplementedError("Rossler model not implemented")
            else:
                raise ValueError(f"Unknown model: {model}")
        else:
            library = [torch.ones(z.shape[0], device=z.device)]
            for i in range(latent_dim):
                library.append(z[:, i])

            if poly_order > 1:
                for i in range(latent_dim):
                    for j in range(i, latent_dim):
                        library.append(z[:, i] * z[:, j])

            if poly_order > 2:
                for i in range(latent_dim):
                    for j in range(i, latent_dim):
                        for k in range(j, latent_dim):
                            library.append(z[:, i] * z[:, j] * z[:, k])

            if poly_order > 3:
                for i in range(latent_dim):
                    for j in range(i, latent_dim):
                        for k in range(j, latent_dim):
                            for p in range(k, latent_dim):
                                library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p])

            if poly_order > 4:
                for i in range(latent_dim):
                    for j in range(i, latent_dim):
                        for k in range(j, latent_dim):
                            for p in range(k, latent_dim):
                                for q in range(p, latent_dim):
                                    library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, q])

            if include_sine:
                for i in range(latent_dim):
                    library.append(torch.sin(z[:, i]))

        return torch.stack(library, dim=1)


class SindyAutoencoder(nn.Module):
    def __init__(self, params):
        super(SindyAutoencoder, self).__init__()
        self.params = params
        self.latent_dim = params['latent_dim']
        self.input_dim = params['input_dim']
        self.widths = params['widths']
        self.activation = params['activation']
        self.library_dim = params['library_dim']
        self.poly_order = params['poly_order']
        
        self.encoder = self.make_network(self.input_dim, self.latent_dim, self.widths, name='encoder')
        self.decoder = self.make_network(self.latent_dim, self.input_dim, self.widths[::-1], name='decoder')

        print(self.params)
        
        if not params['trainable_auto']:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
        
        self.sindy = Sindy(self.library_dim, self.latent_dim, self.poly_order, 
                           model=params['model'], 
                           initializer=params['coefficient_initialization'],
                           actual_coefs=params['actual_coefficients'],
                           rfe_threshold=params['coefficient_threshold'],
                           include_sine=params['include_sine'],
                           exact_features=params['exact_features'],
                           fix_coefs=params['fix_coefs'],
                           sindy_pert=params['sindy_pert'],
                           ode_net=params['ode_net'],
                           ode_net_widths=params['ode_net_widths'])
    
    def make_network(self, input_dim, output_dim, widths, name):
        layers = []
        in_features = input_dim
        for i, w in enumerate(widths):
            layers.append(nn.Linear(in_features, w, bias=(i != 0 or self.params['use_bias'])))
            layers.append(self.get_activation())
            in_features = w
        layers.append(nn.Linear(in_features, output_dim, bias=self.params['use_bias']))
        return nn.Sequential(*layers)
    
    def get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'elu':
            return nn.ELU()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)