import numpy as np
import sys
class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
    
    def softmax(self, o):
        exp_score = np.exp(o)
        s = exp_score / np.sum(exp_score)
        return s
    
    def forward(self, x):
        T = len(x)
        
        s = np.zeros((T+1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((T, self.word_dim))
        
        for t in xrange(T):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = self.softmax(self.V.dot(s[t]))
        return [o, s]
    
    def predict(self, x):
        [o, s] = self.forward(x)
        return np.argmax(o, axis=1)
    
    def calculate_total_loss(self, x, y):
        L = 0
        for i in xrange(len(y)):
            o, s = self.forward(x[i])
            label_one_predictions = o[np.arange(len(y[i])), y[i]]
            L += -1 * np.sum(np.log(label_one_predictions)) 
        return L
    
    def calculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y)/N
    
    def bptt(self, x, y):
        '''
        Be careful with the derivations
        Gradients indices should corresponding to variables in forward procedure
        '''
        T = len(y)
        dLdU = np.zeros(self.U.shape)
        dLdW = np.zeros(self.W.shape)
        dLdV = np.zeros(self.V.shape)
        
        [o, s] = self.forward(x)
        
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)

            delta_s = self.V.T.dot(delta_o[t]) * (1-(s[t]**2))
            
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_s, s[bptt_step-1])
                dLdU[:, x[bptt_step]] += delta_s
                
                delta_s = self.W.T.dot(delta_s) * (1-(s[bptt_step-1]**2))
        return [dLdU, dLdV, dLdW]
    
    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)
     
    def sgd(self, x, y, learning_rate):
        dLdU, dLdV, dLdW = self.bptt(x, y)
        
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
        
    def train(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            if (epoch % evaluate_loss_after==0):
                loss = self.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                print "Loss after num_examples_seen=%d epoch=%d: %f" % (num_examples_seen, epoch, loss)
                
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate *= 0.5
                    print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()
            for i in range(len(y_train)):
                self.sgd(X_train[i], y_train[i], learning_rate=learning_rate)
                num_examples_seen += 1
                
    def generate_sentence(self):
        new_sentence = [word_to_index[sentence_start_token]]
        while not new_sentence[-1] == word_to_index[sentence_end_token]:
            [next_word_probs, _] = self.forward(new_sentence)
            sampled_word = word_to_index[unknown_token]
            
            while sampled_word == word_to_index[unknown_token]:
                # using multinomial to sampling
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
            new_sentence.append(sampled_word)
        sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
        return sentence_str
    
            
