function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       
  % num_labels = No. of output classifier (Here, it is 10)
  % DIMENSIONS:
  % all_theta = 10 x 401 = num_labels x (input_layer_size+1) == num_labels x (no_of_features+1)
  
  prob_mat = X * all_theta';     % 5000 x 10 == no_of_input_image        x num_labels
  [prob, p] = max(prob_mat,[],2); % m  x 1 
  %returns maximum element in each row  == max. probability and its index for each input image
  %p: predicted output (index)
  %prob: probability of predicted output
  
  %%%%%%%% WORKING: Computation per input image %%%%%%%%%
  % for i = 1:m                               % To iterate through each input sample
  %     one_image = X(i,:);                   % 1 x 401 == 1 x no_of_features
  %     prob_mat = one_image * all_theta';    % 1 x 10  == 1 x num_labels
  %     [prob, out] = max(prob_mat);
  %     %out: predicted output
  %     %prob: probability of predicted output
  %     p(i) = out;
  % end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  %%%%%%%% WORKING %%%%%%%%%
  % for i = 1:m
  %     RX = repmat(X(i,:),num_labels,1);
  %     RX = RX .* all_theta;
  %     SX = sum(RX,2);
  %     [val, index] = max(SX);
  %     p(i) = index;
  % end
  %%%%%%%%%%%%%%%%%%%%%%%%%%






% =========================================================================


end
