#!/usr/bin/env python
# coding: utf-8

# In[1]:

#%matplotlib inline
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import numpy.matlib as ml
import scipy.integrate as ig
import scipy.interpolate as ip
import scipy.optimize as optimize
import numpy.random as ran
import os
import glob
#import keras
#import tensorflow as tf
import pandas as pd
import time
import matplotlib.animation as animation



class Density_Model:
    def __init__(self, data_files, tokamak_parameters = [1.0,1.0]):
        
        ##################
        #input file paths as strings in a list
        ##################
        self.data_files = data_files
        #self.coefficients = model_parameters
        self.minor_radius = tokamak_parameters[0]
        self.plasma_current = tokamak_parameters[1]

    def format_data(self):
        #########################
        #this changes the data into readable numpy arrays
        #########################
        self.density_file = self.data_files[0]
        self.rho_file = self.data_files[1]
        self.time_file = self.data_files[2]
        
        ##################################### this is for the density_data
        experiment_density_directory = self.density_file
        experiment_density_raw = pd.read_csv(experiment_density_directory)
        experiment_density = []

        for i in range(0,len(experiment_density_raw)):
            list2 = []
            raw_line = experiment_density_raw.iloc[i][0]
            split_line = raw_line.split()
            for num in split_line:
                list2.append(float(num))
            experiment_density.append(list2)


        ##################################### Radial locations

        experiment_rho_directory = self.rho_file
        experiment_rho_raw = pd.read_csv(experiment_rho_directory)
        experiment_rho = [0.0]
        for i in range(0,len(experiment_rho_raw)):
            raw_points = experiment_rho_raw.iloc[i][0]
            experiment_rho.append(raw_points)



        #################################### Time data
        experiment_time_directory = self.time_file
        experiment_time_raw = pd.read_csv(experiment_time_directory)
        experiment_time = []

        for i in range(0,len(experiment_time_raw)):
            raw_points = experiment_time_raw.iloc[i][0]
            experiment_time.append(raw_points)
            
        if experiment_time[1] > 10: #checks if the data is in ms or s
            for i in range(0,len(experiment_time)):
                experiment_time[i] = experiment_time[i]/1000
        else:
            a = 1
            
            
        #####defines some variables to be called on later####
        self.original_density_data = experiment_density
        self.original_rho_data = experiment_rho
        self.original_time_data = experiment_time
        return
        
        
    def reduce_data(self, normalized = False, reduction_number = 1):

        dens_data = self.original_density_data
        space_data = self.original_rho_data
        time_data = self.original_time_data

        reduced_dens_data = []
        reduced_space_data = []
        for i in range(0,len(dens_data)):
            for j in range(0,len(dens_data[i])):
                if j % reduction_number == 0:
                    reduced_dens_data.append(dens_data[i])
        for i in range(0,len(dens_data[0])):
            reduced_space_data.append(space_data[i])
        reduced_data = [reduced_dens_data, reduced_space_data, time_data]
        self.reduced_data = reduced_data #this holds the reduced data
        return
    
    def expand_time_scale(self,extra_time_factor):
        old_simulation_time = []
        for i in range(0,len(self.original_time_data)-1):
            extra_time = list(np.linspace(self.original_time_data[i],self.original_time_data[i+1], 
                                          extra_time_factor))
            
            simulation_time = old_simulation_time + extra_time
            simulation_time.pop()
            old_simulation_time = simulation_time
            
        simulation_time.append(self.original_time_data[-1])
        self.simulation_time = simulation_time
        
    def reduce_time_scale(self, extra_time_factor):
        original_size_n_e = []
        #reduce the simulaton back to its original time scale
        for i in range(0,len(self.full_modeled_density_data)):
            if i % (extra_time_factor - 1) == 0:
                original_size_n_e.append(self.full_modeled_density_data[i])
            else: 
                continue
        self.modeled_density_data = original_size_n_e  #this is the original size as the data
        return self.modeled_density_data
    
    def interpolate_reduced_data(self):
        
        ###################################
        #reinterpolates the reduced and modeled data to return a profile the same size as the original
        ###################################
        dens_data = self.modeled_density_data
        space_data = self.original_rho_data
        time_data = self.original_time_data

        reduced_dens_data = self.modeled_density_data
        reduced_space_data = self.reduced_data[1]

        interp_dens_profile = []
        for i in range(0,len(time_data)):
            interp_dens = ip.InterpolatedUnivariateSpline(space_data,dens_data[i])
            interp_dens_profile.append([])
            for j in range(0,len(space_data)):
                interp_dens_profile[i].append(interp_dens(space_data[j]))

        self.interp_data = [interp_dens_profile,space_data,time_data]
        return
    
    def gradient(self, quantity, spaces):
        ###########
        #standard linear gradient
        ###########
        grad = np.array(quantity)
        for i in range(1,len(quantity)-1):
            step = (spaces[i+1] - spaces[i-1])
            grad[i] = (quantity[i+1] - quantity[i-1]) / step
        #makes the gradient at the edges the same as the value for the point right next to them
        grad[0] = grad[1]
        grad[-1] = grad[-2]
        return grad
        
        
    def get_gamma_2block(self, grad_n_e, n_e, spots, Ds, Vs):  #this is the place to mess with the Ds and Vs ##### Right now Ds and Vs are two each
        grad_n_e = np.array(grad_n_e,dtype=float)
        n_e = np.array(n_e,dtype=float)

        gamma = np.array(n_e,dtype = float)

        for i in range(0,len(n_e)):
            if spots[i] >= 0.9:    ############## Here controls where the second block begins and it changes D and V values
                D = Ds[1]
                V = Vs[1]
            else:
                D = Ds[0]
                V = Vs[0]
            gamma[i] = -1*D*grad_n_e[i] + V*n_e[i]
        return np.array(gamma,dtype=float)
    
    def source_func(self, spaces, height, width): #add a new source as an exponential, not a gaussian
        source_profile = np.zeros_like(spaces)
        for i in range(0,len(spaces)):
            source_profile[i] = height*np.exp(-1*width*(spaces[i]-1)**2) + 0.1*height
                                #gaussian term                            #complete upward shift 10% of edge sourcing
        
        return np.array(source_profile,dtype=float)
    
    
    def diff_eq (self, grad_gamma, source):
        dn_dt = -1*grad_gamma + source
        return np.array(dn_dt,dtype=float)
    
    def model_density(self): # this is going to be like model3 but it will use interpolation and find the gradient that way
        ##### need to make extra time points to avoid numerical instability #####
        extra_time_factor = 15
        self.expand_time_scale(extra_time_factor)
        
        ############### Define needed variables ##############
        
        y0 = self.original_density_data[0] #, self.reduced_data[0][0] 
        spaces = self.original_rho_data #self.reduced_data[1]
        t = self.simulation_time # generated by the expand time scale function
        
        #print(t)
        source_height = self.coefficients[0]
        source_width = self.coefficients[1]
        D1 = self.coefficients[2]
        D2 = self.coefficients[3]
        V1 = self.coefficients[4]
        V2 = self.coefficients[5]

        D_set = [D1,D2]
        V_set = [V1,V2]
        source_coefficients = [source_height, source_width]
        
        ##################### This begins the actual function ###############
        f = self.diff_eq
        source = self.source_func(spaces, source_height, source_width)
        gradient = self.gradient
        get_gamma = self.get_gamma_2block
        rstep = spaces[1] - spaces[0] 
        #gstep = 0.01 

        n_e = [] #this just makes a list that can then be filled as the model goes on
        for i in t:
            n_e.append(np.array([1]))
        n_e[0] = np.array(y0,dtype=float)

      ################### This is the big time loop ###############
        for j in range(0,len(t)-1):
            #this is the full time step size
            h = 0.1*(t[j+1] - t[j])
            n_e0 = np.array(n_e[j])

            #####calc all the stuff for k1 ######
            
            n_e1 = n_e0
            grad_n1 = gradient(n_e1, spaces)
            gam1 = get_gamma(grad_n1,n_e1,spaces,D_set, V_set)
            grad_gam1 = gradient(gam1,spaces)
            
            #k1 is slope at start*timestep
            k1 = h * f(grad_gam1,source)

            #####calc all stuff for k2#####
            
            n_e2 = n_e1
            for i in range(0,len(n_e1)):
                n_e2[i] = n_e0[i] + (k1[i] / 2)

            grad_n2 = gradient(n_e2,spaces)
            gam2 = get_gamma(grad_n2, n_e2, spaces, D_set, V_set)
            grad_gam2 = gradient(gam2,spaces)

            #k2 is (slope at half time step)*(starting density+ k1 correction)
            k2 = h*f(grad_gam2,source)

            #####calc all stuff for k3#####
            n_e3 = n_e2
            for i in range(0,len(n_e1)):
                n_e3[i] = n_e0[i] + (k2[i] / 2)
            #print(n_e3)    
            grad_n3 = gradient(n_e3,spaces)
            gam3 = get_gamma(grad_n3,n_e3, spaces, D_set, V_set)
            grad_gam3 = gradient(gam3,spaces)

            #k3 is slope part way
            k3 = h*f(grad_gam3,source)

            #####calc all stuff for k4######
            
            n_e4 = n_e3
            for i in range(0,len(n_e1)):
                n_e3[i] = n_e0[i] + k3[i] 

            grad_n4 = gradient(n_e4, spaces)
            gam4 = get_gamma(grad_n4, n_e4, spaces, D_set, V_set)
            grad_gam4 = gradient(gam4,spaces)

            #k4 is slope at the end
            k4 = h*f(grad_gam4,source)

            ############## calculate k and andvance the scheme ###########
            
            k = (k1 + 2*k2 + 2*k3 + k4)/6
            n_e[j+1] = n_e[j] + k
            
            for i in range(0,len(n_e[j+1])):
                if n_e[j+1][i] < 0:
                    n_e[j+1][i] = 0
                else:
                    continue

        self.full_modeled_density_data = n_e # this is the full simulation with extra time steps
        self.modeled_data = self.reduce_time_scale(extra_time_factor)
        
        return [self.modeled_data, self.original_rho_data, self.original_time_data]
    

    def model(self,model_parameters):
        self.coefficients = model_parameters
        self.format_data()
        #self.reduce_data()
        modeled_data =self.model_density()
        #self.interpolate_reduced_data()
        return modeled_data
    
########################################################################################

class Optimization:
    def __init__(self, comparison_data,model_data):
        self.dens_data = comparison_data
        self.DM = Density_Model(model_data)
        
    def chi2_difference(self, params, normalize = False):
        data = self.dens_data
        diff = 0
        trial = self.DM.model(params)[0]
        if normalize == False:
            normalization = 1
        elif normalize == True:
            normalization = 1e20*0.9/(np.pi*0.67**2)
        
        for i in range(0,len(data)):
            for j in range(0,len(data[i])):
                diff += ((data[i][j] - trial[i][j])/normalization)**2
        return diff
    
    def chi2_difference_GD(self, params):
        if len(params) == 6:
            params_modified = params
        else:
            params_modified = []
            for i in range(0,6-len(params)):
                params_modified.append(0)
            for i in range(0,len(params)):
                params_modified.append(params[i])
            
        data = self.dens_data
        diff = 0
        trial = self.DM.model(params_modified)[0]
        for i in range(0,len(data)):
            for j in range(0,len(data[i])):
                diff += (data[i][j] - trial[i][j])**2
        return diff
    
    def optimize(self, initial_guess, iterations = 500, optimization_bounds = None):
        self.optimization_bounds = optimization_bounds
        self.optimized_values = optimize.minimize(self.chi2_difference, initial_guess, args=(), method='nelder-mead', 
                                                  options = {'maxiter': iterations , 'adaptive': True })
        return self.optimized_values
    
    def optimize_Powell(self,initial_guess, iterations = 500, optimization_bounds = None):
        self.optimization_bounds = optimization_bounds
        self.optimized_values = optimize.minimize(self.chi2_difference, initial_guess, args=(), method='Powell', 
                                                  options = {'maxiter': iterations})
        return self.optimized_values
    
    def partial(self, var_index, func):
        f = func
        ''' 
        Returns a function object to compute the partial derivative of f with respect to x[i].
        
        f(x) is assumed to be a scalar function of a vector or scalar argument x.
        '''
        def df(x, f = f, i = var_index,):
            x = np.array(x, dtype = np.float64) # make a copy and assure the use of 64-bit floats
            h = self.step_list[i]
            x[i] += h
            f_plus = f(x)
            x[i] -= 2*h
            f_minus = f(x)
            return (f_plus - f_minus) / (2.0*h)
    
        # note, partial() returns a function object, not the result of the function
        return df


    def opt_gradient (self,x):
        f = self.f
        gradient_vector = []
        for i in range(0,len(x)):
            df = self.partial(i,f)
            gradient_vector.append(df(x))
        gradient_vector = ml.matrix(gradient_vector) # This is a 1x6 row vector
        return gradient_vector
    
    
    def opt_hess(self,x):
        f = self.f
        gradient = []
        hessian = []
        
        for i in range(0,len(x)):
            df = self.partial(i,f)
            gradient.append(df)
        #print('gradient is', gradient[:])
        
        
        for i in range(0,len(x)):
            hessian_row = []
            #hessian.append(gradient)
            for j in range(0,len(x)):
                ddf = self.partial(i, gradient[j])
                #print(type(hessian[i]))
                #print(type(ddf))
                #print(i,j)
                hessian_row.append( ddf(x) )
                #print(hessian_row[j])
            hessian.append(hessian_row)
                
        hessian_matrix = ml.matrix(hessian)
        return hessian_matrix


    def optimize_GD(self, init_guess, maxiter = 15, tolerance = 10):
        self.step_list = [0.01,0.01,0.01,0.01,0.001,0.01]
        self.f = self.chi2_difference_GD
        #self.dens_data = data[0]
        x = init_guess
        iterations = 0
        accuracy = 1e10
        while (accuracy > tolerance) and (iterations < maxiter): # loops until sufficiently accurate or hits max interations

            ##### calculate needed matricies and gradients ##### 
            grad_f = self.opt_gradient(x) #calculates gradient vector
            #print(grad_f)
            hess = self.opt_hess(x) # calculates hessian matrix
            print(hess)
            inv_hess = hess.I # inverts hessian matrix
            x_vec = ml.matrix(x)
            x_new = x_vec - (grad_f * inv_hess) # sets new coordinate to check


            ##### calculates the accuracy of the guess  for termination #####
            # accuracy is based on how much you moved from last guess
            diff_vector = x_new - x_vec
            #print(diff_vector)
            #print(time.time() - start_time)
            self.diff_v = diff_vector
            acccuracy = 0
            for i in range(0,len(init_guess)):
                accuracy += diff_vector.A1[i]
            iterations += 1

            ##### Makes the new point the old guess before repeating the loop #####
            x = x_new.A1

        #print(iterations)
        #print(accuracy)
        return x
    
###############################################################################################

class Visualization:
    
    def __init__(self,data):
        self.fig, self.ax = plt.subplots()
        ##### sets some variables for use in the animations and plotting #####
        self.viz_dens_data = data[0]
        self.viz_rho_data = data[1]
        self.viz_time_data = data[2]
        
    def plot_dens_v_rho(self,time_index, ttl = "title"):
        time_plt = str(self.viz_time_data[time_index])
        plt.xlabel("$\\rho$")
        plt.ylabel("$n_e$")
        
        if ttl == "title":
            plt.title("Density vs $\\rho$ at t=" + time_plt)
        else:
            plt.title(ttl)
        plt.plot(self.viz_rho_data, self.viz_dens_data[time_index])
        

    def plot_dens_v_time(self,rho_index, ttl = "title"):
        rho_plt = str(self.viz_rho_data[rho_index])
        
        time_dat = []
        for i in range(0,len(self.viz_dens_data)):
            time_dat.append(self.viz_dens_data[i][rho_index])
                
        plt.plot(self.viz_time_data, time_dat)
        plt.xlabel("Time")
        plt.ylabel("$n_e$")
        if ttl == "title":
            plt.title("Density vs Time at $\\rho$ =" + rho_plt)
        else:
            plt.title(ttl)
        

    def animate_dens_v_rho(self,ttl = "title"): #currently plots a static image
        ##### This function animates density vs rho plots over time #####
        
        
        ##### sets up the figure #####
        plt.xlim(self.viz_rho_data[0],self.viz_rho_data[-1])
        plt.ylim(0,6e19)
        plt.xlabel("$\\rho$")
        plt.ylabel("$n_e$")
        plt.title(ttl)

        x = self.viz_rho_data
        line, = self.ax.plot(self.viz_rho_data, self.viz_dens_data[0])

        def animate(i):
            line.set_ydata(self.viz_dens_data[i])  # update the data.
            return line,

        ##### runs the animation #####
        ani = animation.FuncAnimation(
            self.fig, animate, frames = len(self.viz_time_data), interval=40, blit=True, save_count=50)
        return ani
        plt.show()
    
    
    def animate_dens_v_time(self,ttl = "title"): #currenlty plots a static image
        ##### This function animates denisty vs time plots while changinf rho in real time #####
        
        
        ##### sets up the figure #####
        plt.xlim(self.viz_time_data[0],self.viz_time_data[-1])
        plt.ylim(0,6e19)
        plt.xlabel("time")
        plt.ylabel("n_e")
        plt.title(ttl)

        x = self.viz_time_data
        first =[]
        for i in range(0,len(self.viz_dens_data)):
            first.append(self.viz_dens_data[i][0])
        line, = self.ax.plot(x, first)


        def animate(i):
            dat = []
            for j in range(0,len(self.viz_dens_data)):
                dat.append(self.viz_dens_data[j][i])
            line.set_ydata(dat)  # update the data.
            return line,

        ##### runs the animation #####
        ani = animation.FuncAnimation(
            self.fig, animate, frames = len(self.viz_rho_data), interval=40, blit=True, save_count=50)
        return ani
        plt.show()

#############################################################################################

class ML_Data_Preparation:
    
    def __init__(self,temporal_data_size, restrictions = None):
        self.temporal_data_size = temporal_data_size
        
        height_opt , decay_opt, D_opt, V_opt = 2, 2, 6, 21
        height_max, decay_max, D_max, V_max = 1, 1, 5, 10
        height_min, decay_min, D_min, V_min = 0, 0, 0, -10
        
        self.source_heights = np.linspace(decay_min, decay_max, height_opt)
        self.source_decay = np.linspace(decay_min, decay_max ,decay_opt)
        self.Ds = np.linspace(D_min, D_max, D_opt)
        self.Vs = np.linspace(V_min, V_max, V_opt)
        
        
        ##### This sets the output options for the NN #####
        self.param_options = []
        restrictions_met = True
        for a in self.source_heights:
            for b in self.source_decay:
                for c in self.Ds:
                    for d in self.Ds:
                        for e in self.Vs:
                            for f in self.Vs:
                        
                                if restrictions_met == True:
                                    # This index corresponds to dense layer output node
                                    self.param_options.append((a,b,c,d,e,f)) 
                                else:
                                    continue
        #print('total_param_options: ', len(self.param_options))
        
    def prepare_data(self, data, normalization_parameters = [9.9e5,0.67]):  # only preps one density profile
        #data is a list of density profiles, data should be a 3d list
        
        ##### transforms many density profiles to a vector for the NN #####
        prepared_data = []
        for i in range(0,self.temporal_data_size): # the length only goes as high as the size of the model will accept, 1002
            data_vector = []
            for j in range(1,len(data[i])-1): # the bounds cut off the radial edge points to fit the NN
                data_vector.append(data[i][j])
            prepared_data.append(data_vector)
        
        ##### normalizes density profiles based on Greenwald Density limit #####
        norm = 1e20*normalization_parameters[0]/(np.pi*normalization_parameters[1]**2)
        self.Dens_limit = norm
        
        scaled_prepared_data = []
        for i in range(0,len(prepared_data)):
            scaled_vec = []
            for j in range(0,len(prepared_data[i])):
                scaled_vec.append([prepared_data[i][j]/norm])
            scaled_prepared_data.append(scaled_vec)
            
            
        self.prepared_data = scaled_prepared_data # needs to be a numpy array    
        return scaled_prepared_data   
        
        
    def generate_fake_training_data(self,num_samples):
        
        ##### create options for random choices of transport coefficients #####
        
        parameter_options = [self.source_heights,self.source_decay,self.Ds,self.Ds,self.Vs,self.Vs]
        
        ########## create a (fake) training data set #########
        fake_training_set = [] # holds the input training data
        fake_training_answers = [] # holds index of model parameters
        fake_training_parameters = [] # holds the model parameters
        
        for data_point in range(0,num_samples): # this is the big loop to generate all the training data
            param_index = ran.choice(range(0,len(self.param_options)),1)[0]
            training_parameters = self.param_options[param_index]
                
            training_sample_data = ["158348_density-Copy1", "158348_rho-Copy1", "158348_time_values-Copy1"] # data to start model off of

            DM = Density_Model(training_sample_data)
            training_data_point = DM.model(training_parameters)

            preped_fake_data_point = self.prepare_data(training_data_point[0]) # preps the data to be the right shape for NN
            
            fake_training_set.append(np.asarray(preped_fake_data_point))
            fake_training_answers.append(np.asarray(param_index))
            fake_training_parameters.append(np.asarray(training_parameters))
                #each data point is a len(time)xlen(rho) list, list is num_samples long
            #fake_training_answers.append(np.asarray(ran.choice([0,1],1)[0]))  
                
                
        ##### new variables of the class instance for later use #####        
        self.fake_training_data = fake_training_set
        self.fake_training_parameters = fake_training_answers
        
        return fake_training_set, fake_training_answers

####################################################################################################    

class ML_Building_and_Training:
    def __init__(self):
        #######################################
        ##### Design aspects of the model #####
        #######################################


        ##### These are the set dimensions the model expects to see #####

        self.spacial_data_size = 99 # so its divisible by 3, drop edgemost point and inner most point
        self.temporal_data_size = 120 # so its divisible by 3
    
    def build_data_sets(self, training_size = 90, validation_size = 10):
        
        ##################################################
        #### build training data and validation data #####
        ##################################################

        ##### this makes the validation and traiing data sets #####
        MLDP = ML_Data_Preparation(120)
        train_dataset, train_data_labels = MLDP.generate_fake_training_data(training_size) #training data
        validation_dataset, validation_data_labels = MLDP.generate_fake_training_data(validation_size) #validation data

        ##### converts data sets to a format for the NN #####
        train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset, train_data_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((validation_dataset, validation_data_labels))


        ##### Shuffles training data around  to remove ordering biases #####
        BATCH_SIZE = 5
        SHUFFLE_BUFFER_SIZE = 1

        self.train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        self.test_dataset = test_dataset.batch(BATCH_SIZE)

        ##### sets the output options for the NN #####
        self.output_options = MLDP.param_options
        self.output_len = len(self.output_options)
        
    def build_NN(self):
        ###############################
        ##### This Neural Netowrk #####
        ###############################


        # first dense line is the first hidden layer
        # filters is the number of neurons
        #input shape determines the shape of the input, 1 being a vector
        #last dense layer is output layer and number of neurons corresponds to the possible outputs

        self.Seq_NN = Sequential([
                             Conv2D(filters=32, kernel_size=(3), activation='relu', padding='same',input_shape=(self.temporal_data_size, self.spacial_data_size,1)),
                             MaxPooling2D(pool_size=(2), strides=2),
                             Conv2D(filters=32, kernel_size=(3), activation='relu', padding='same'),
                             MaxPooling2D(pool_size=(2), strides=2),
                             Conv2D(filters=32, kernel_size=(3), activation='relu', padding='same'),
                             MaxPooling2D(pool_size=(2), strides=2),
                             Flatten(),
                             Dense(units= self.output_len ,activation='softmax'),
        ])

        # Gives a summary of what the model looks like
        self.Seq_NN.summary() 
        
    def train_NN(self):
        ##########################
        ##### Model Training #####
        ##########################

        self.Seq_NN.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = self.Seq_NN.fit(self.train_dataset, epochs=10, 
                            validation_data=(self.test_dataset))


# In[ ]:

