import numpy as np, matplotlib.pyplot as plt, pandas as pd
from scipy.stats import pearsonr

class data_prepros():

    def loadFile(self, df):
        # load a comma-delimited text file into an np matrix
        resultList = []
        f = open(df, 'r')
        for line in f:
            line = line.rstrip('\n')  
            sVals = line.split(',')   
            fVals = list(map(np.float32, sVals))  
            resultList.append(fVals) 
        f.close()
        return np.asarray(resultList, dtype=np.float32)
        # end loadFile

    def SplitData(self, data, Inputsize, Outputsize):
        x_values = np.zeros(shape = [len(data),Inputsize],dtype = np.float32)
        d_values = np.zeros(shape = [len(data),Outputsize],dtype = np.float32)

        for i in range(len(data)):
            for j in range(Inputsize):
                x_values[i,j] = data[i,j]
            for j in range(Outputsize):
                d_values[i,j] = data[i,Inputsize+j]

        x_values = x_values.T
        d_values = d_values.T

        return x_values,d_values

class deep_NN():

    def normalize(self, X):

        feat_count = X.shape[0]
        M = X.shape[1]
        m = np.zeros(shape = [feat_count, 1],dtype = np.float32)
        s = np.zeros(shape = [feat_count, 1],dtype = np.float32)
        n = np.zeros(shape = [feat_count, M],dtype = np.float32)
        for i in range(feat_count):
            m[i] = np.mean(X[i])
            s[i] = np.std(X[i])
        for i in range(feat_count):
            n[i] = (X[i] - m[i])/s[i]

        return n

    def sigmoid(self, z):
        """
        Compute the sigmoid of z

        Arguments:
        -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        cache -- z
        """
        s = 1/(1+np.exp(-z))
        cache = z

        return s, cache

    def sigmoid_backward(self, dA, activation_cache):
        
        Z = self.sigmoid(activation_cache)[0]
        
        dZ = np.multiply(dA, (Z - np.square(Z)))
        
        return dZ

    def relu(self, z):
        """
        Compute the sigmoid of z

        Arguments:
        -- A scalar or numpy array of any size.

        Return:
        s -- relu(z)
        cache -- z
        """
        s = z * (z > 0)
        cache = z

        return s, cache

    def relu_backward(self, dA, activation_cache):
        
        z = activation_cache
        z[z <= 0] = 0
        z[z > 0] = 1

        dZ = np.multiply(dA, z)
        
        return dZ

    def initialize_parameters_deep(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
    
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
        """
        parameters = {}
        L = len(layer_dims)            # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros([layer_dims[l], 1])
        
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
        return parameters

    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
    
        Z = np.dot(W, A) + b
    
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
    
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
        """
        cache = []
    
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
    
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
    
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
    
        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network
    
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
            caches.append(cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
        caches.append(cache)

        out_dim = parameters['W' + str(L)].shape[0]

        assert(AL.shape == (out_dim, X.shape[1]))
            
        return AL, caches

    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (LabelsDim, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (LabelsDim, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
    
        m = Y.shape[1]

        # Compute loss from aL and y.
        cost = -(1/m)*np.sum(np.multiply(Y, np.log(AL))+np.multiply((1 - Y), np.log(1 - AL)), axis = 1, keepdims = True)

        #cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        #assert(cost.shape == ())
    
        return cost

    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (1/m)*np.dot(dZ, A_prev.T)
        db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
    
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
    
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
    
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
    
        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
        Returns:
        grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
        # Initializing the backpropagation
        dAL =  - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation="sigmoid")
    
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
    
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
    
        Returns:
        parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
        """
    
        L = len(parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
        return parameters

    def update_learning_rate(self, cost):
        
        learning_rate = np.square(np.sum(cost) / len(cost))
        
        return learning_rate

    def L_layer_model(self, X, Y, layer_dims, learning_rate =0, num_iterations = 3000, print_cost=False):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
        Arguments:
        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
    
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        costs = {}

        for x in range(1, layer_dims[len(layer_dims)-1] + 1):
            costs["Label " + str(x)] = []                      # keep track of cost
    
        # Parameters initialization. (≈ 1 line of code)
        parameters = self.initialize_parameters_deep(layer_dims)

        count = 0
        l = []

        # Loop (gradient descent)
        for i in range(0, num_iterations):
            
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self.L_model_forward(X, parameters)

            # Compute cost.
            cost = self.compute_cost(AL, Y)
            
            # Backward propagation.
            grads = self.L_model_backward(AL, Y, caches)

            # Determine num_iteration
            #num_iterations = np.square(layers_dims[0]) * cost[0]

            # Determine learning_rate
            if learning_rate == 0 or count >= 1:
                learning_rate = self.update_learning_rate(cost)
                count += 1
            l.append(learning_rate)

            # Update parameters.
            parameters = self.update_parameters(parameters, grads, learning_rate)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                for x in range(1, layer_dims[len(layer_dims)-1] + 1):
                    print("Cost of %i no label after iteration %i: %f" %(x, i, np.squeeze(cost[x-1])))
                    costs["Label " + str(x)].append(np.squeeze(cost[x-1]))
                print("\n")
                

        # plot the cost
        for x in range(1, layer_dims[len(layer_dims)-1] + 1):
            plt.plot(costs["Label " + str(x)])
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title('learning rate = ' + str(learning_rate))
        plt.show()

        plt.plot(l)
        plt.ylabel('learning rate')
        plt.xlabel('iterations')
        plt.title('Last learning rate = ' + str(learning_rate))
        plt.show()

        return parameters

    def auto_model(self, X, Y, learning_rate =0, num_iterations = 3000, print_cost=True):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
        Arguments:
        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
    
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        
        layer_dims = []
        c = 0
        count = 0
        node_max = X.shape[0]
        node = 1
        l = []

        costs = {}
        for x in range(1, Y.shape[0] + 1):
            costs["Label " + str(x)] = []                      # keep track of cost

        while c <= 0 and node <= node_max:
            
            layer_dims = [X.shape[0], node, Y.shape[0]]

            # Parameters initialization. (≈ 1 line of code)
            parameters = self.initialize_parameters_deep(layer_dims)

            for i in range(0, 1500):
            
                # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                AL, caches = self.L_model_forward(X, parameters)
        
                # Compute cost.
                cost = self.compute_cost(AL, Y)
            
                # Backward propagation.
                grads = self.L_model_backward(AL, Y, caches)

                # Determine num_iteration
                #num_iterations = np.square(layers_dims[0]) * cost[0]

                # Determine learning_rate
                if learning_rate == 0 or count >= 1:
                    learning_rate = self.update_learning_rate(cost)
                    count += 1
                l.append(learning_rate)

                # Update parameters.
                parameters = self.update_parameters(parameters, grads, learning_rate)

            if max(cost) <= .4:
                c += 1
            else:
                node += 1

        # Loop (gradient descent)
        for i in range(0, num_iterations):
            
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self.L_model_forward(X, parameters)
        
            # Compute cost.
            cost = self.compute_cost(AL, Y)
            
            # Backward propagation.
            grads = self.L_model_backward(AL, Y, caches)

            # Determine num_iteration
            #num_iterations = np.square(layers_dims[0]) * cost[0]

            # Determine learning_rate
            learning_rate = self.update_learning_rate(cost)
            l.append(learning_rate)

            # Update parameters.
            parameters = self.update_parameters(parameters, grads, learning_rate)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                for x in range(1, layer_dims[len(layer_dims)-1] + 1):
                    #print("Cost of %i no label after iteration %i: %f" %(x, i, np.squeeze(cost[x-1])))
                    costs["Label " + str(x)].append(np.squeeze(cost[x-1]))
                #print("\n")
                

        # plot the cost
        for x in range(1, layer_dims[len(layer_dims)-1] + 1):
            plt.plot(costs["Label " + str(x)])
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title('learning rate = ' + str(learning_rate))
        plt.show()

        plt.plot(l)
        plt.ylabel('learning rate')
        plt.xlabel('iterations')
        plt.title('Last learning rate = ' + str(learning_rate))
        plt.show()

        return layer_dims, parameters

    def validation(self, X, Y, parameters):

        AL = self.L_model_forward(X, parameters)[0]
        a = np.asarray(AL).T
        AL_max = np.amax(a, axis=1, keepdims=True)
        m = AL_max.shape[0]
        count = 0

        for i in range(m):
            a[i][a[i] < AL_max[i]] = 0
            a[i][a[i] >= AL_max[i]] = 1 
        
        count = np.sum(np.multiply(a, Y.T))

        acc = count / m

        return acc

    def predict(self, X, parameters):

        AL = self.L_model_forward(X, parameters)[0]
        a = np.asarray(AL).T
        label_class = {}
        AL_max = np.amax(a, axis=1, keepdims=True)
        m = AL_max.shape[0]

        for i in range(m):
            a[i][a[i] < AL_max[i]] = 0
            a[i][a[i] >= AL_max[i]] = 1
        
        a = a.tolist()
        
        for i in range(m):
                label_class["label for " + str(i+1) + " no example"] = a[i].index(1)

        return label_class

    # def corr_model(self, X, Y):

    #     corr = np.corrcoef(X)
    #     label_count = Y.shape[0]

    #     layer = 1
    #     for i in range(layer):
    #         node[i] = 1
    #     return corr
        
        
