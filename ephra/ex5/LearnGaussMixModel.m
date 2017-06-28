
% Implementation of a split-based EM-algorithm
%---------------------------------------------
%
% Inputs:
%   trainVect: array of size n_featurevects x n_dimensions
%   n_comp: number of desired components in the model
%
% Outputs: 
%   model: structure with estimated model
%   model.weight: (n_components x 1) vector with weight for each component
%   model.mean: (n_components x n_dims=3) matrix with mean vectors for each
%                component --> mean of vector i: model.mean(i,:)
%   model.covar: (n_components x n_dims=3 x n_dims=3) matrix with
%                covariance matrices for each component
%                squeeze(model.covar(i,:,:)) returns the i'th covar matrix
%
% main function of the EM-algorithm
function model = LearnGaussMixModel(trainVect, n_comp)

    % initialization of the model using a structure
    % at the starting point the algorithm will always be initialized with 
    % one cluster 
    whos trainVect
    model.weight(1,:)=1;
    model.mean=[0,0,0];
    model.covar(1,:,:)=[1 0 0; 0 1 0; 0 0 1];
    %model.covar=[1 0 0; 0 1 0; 0 0 1];
    
    % threshold for stopping the iteration
    eps=10^-6;
    
    % loop over the desired number of components
    for i=1:n_comp
        
        % the first overall model probability is -infinity
        LastPX=-inf;
        
        % calculate the logarithmic overall model probability
        LnTotalProb = CalcLnTotalProb(model, trainVect);
        
        % while the threshold is bigger than the difference of the overall
        % probabilities from this and the last iteration...
        while (LnTotalProb-LastPX>eps)
            LastPX = LnTotalProb;
            
            % E-step:
            % compute for each feature vector the probabilities for each
            % component
            LnCompProb = GmmEStep(model, trainVect);
            
            % M-step:
            % Maximize the model by reestimating the model parameters using
            % the probabilities of E-step
            model = GmmMStep(model, trainVect, LnCompProb);
            
            % again compute the overall probability of the model
            % since EM always converges, this value is always higher than
            % in the last iteration
            LnTotalProb = CalcLnTotalProb(model, trainVect);      
        end
        
        % clear current figure window
        clf
        % plot the estimated GMM
        PlotGMM(model,trainVect);
        % flush
        drawnow;
        
        % find a component to split into two and init them
        % but only, if the desired number n_comp is not reached
        if i < n_comp
            model = InitNewComponent(model, trainVect);
        end
    end
end


%--------------------------------------------------------------------------
% logarithmic probability of all vectors for all components
function LnVectorProb = CalcLnVectorProb(model, trainVect)
   % IMPLEMENT THIS FUNCTION (TASK A.a)
   % c   - number of clusters ?!
   % k   - ???
   % ??   - mean vector
   % sigma   - covariance matrix
   % |sigma| - det(covar matrix)
   % sigma-1 - inverse of covariance matrix
   % x_i - element of training vec
   % alpha - weight?!

   n_comp = numel(model.weight);
   LnVectorProb = zeros(n_comp,size(trainVect, 1));
   for i = 1:n_comp
     alpha = model.weight(i,:);
     % whos alpha;
     sigma = model.covar(i,:,:);
     % whos sigma
     det_sig = det(squeeze(sigma));
     %whos det_sig
     my = model.mean(i,:);
     whos model.mean
     
     for k = 1:size(trainVect,1)
      x = trainVect(k,:);
      
      A = (x - my)* inv(squeeze(sigma));
      result = log(alpha) - 1/2 * (log(det_sig) + A * (x-my)');
     
      LnVectorProb(i,k) = result;
     end
    
   end
   whos LnVectorProb;
   
end

%--------------------------------------------------------------------------
% E-Step:
% calculation of the probabilities of each feature vector wrt all existing
% components using the current model parameters
function LnCompProb = GmmEStep(model, trainVect)

  n_comp = numel(model.weight);
  M = CalcLnVectorProb(model, trainVect);

  s = zeros(1, size(trainVect, 1));
  s = sum(M,1);

  LnCompProb = zeros(n_comp, size(trainVect, 1));
  for i = 1:n_comp
    LnCompProb(i,:) = M(i,:) / s;
  end

end

%--------------------------------------------------------------------------
% M-Step:
% Estimation of new model parameters according the calculated probabilities
% of the E-Step
function model = GmmMStep(model, trainVect, LnCompProb)

  n_comp = numel(model.weight);

  num_p_in_c = exp(LnCompProb);

  N     = zeros(1,n_comp);
  alpha = zeros(1,n_comp);
  my    = zeros(1,3);
  sigma = zeros(3,3);

  % for each cluster eat a banana and drink some pee
  for i = 1:n_comp
    
    % set number of feature points
    N(1,i) = sum(num_p_in_c,2);
    
    % new weight
    alpha(1,i) = N(1,i) / size(trainVect,1);

    % calculate new mean vecs
    for j = 1:size(trainVect,1)
      rgb = trainVect(j,:);
      my(i,:) = my(i,:) + rgb * num_p_in_c(i, j);
    end
    my(i,:) = 1 / N(1,i) * my(i,:);

    % covar calc
    for j = 1:size(trainVect,1)
      rgb = trainVect(j,:);
      sigma(i,:) = (rgb - my(i,:)) * (rgb - my(i,:))' * num_p_in_c(i, j);
    end
    sigma(i,:) = 1 / N(1,i) * sigma(i,:);
  
    % set new model
    model.weight(i,:) = alpha;
    model.mean(i,:)=my;
    model.covar(i,:,:) = sigma;

  end
  

end

%--------------------------------------------------------------------------
% calculation of the global probability given the current model and the
% feature vectors
function LnTotalProb = CalcLnTotalProb(model, trainVect)
    
    % get the current number of components in the model
    n_comp = numel(model.weight); 
    %n_comp = numel(model.weight(1,:)); %  Ver??ndert von ephra !!!!!!!!

    % logarithmic probability for all vectors in all components
    LnVectorProb = CalcLnVectorProb(model, trainVect);
        
%     % the log of a sum cannot easity be computed from single log-values!
%     % so for this step we have to use the exp-function and afterwards take
%     % the log of the sum! (log of a product-->sum log values, but there is 
%     % no such rule for log of a sum!)
%     s = sum(exp(LnVectorProb),1);
%     LnTotalProb = sum(log(s));

    % the result abuve could be wrong tue to very small values after exp...
    % to be safe, we can compute LnTotalProb using a scaling factor:
    %---------------------------------------------------------------------
    % use scaling factor c = max of log values
    % wrt a feature vector --> one scale factor for each feature vctor

    % max probability for each feature vector
    % [ the maximum probability tells us, to which cluster each vector
    % belongs]
    % this value is used for scaling the probabilities in order to avoid
    % numerical problems for computation of the sum
    max_LnVectorProb = max( LnVectorProb,[],1 );
    
    % resize this array to size of LnTotalProb
    scaling_factors = repmat(max_LnVectorProb, n_comp, 1);
    
    % scaling of logarithmic probabilities before using exp in order to
    %  avoid numerical problems:
    % 1) subtract scaling_factors from LnVectorProb (scaling)
    % 2) take exp of the result (should be no problem after scaling)
    % 3) summarize the n_comp values for each feature vector (as desired)
    % 4) take the logarithm of the sums
    % 5) add the maximum to the result of 4 ("unscaling")
    LnVectorProb_new = max_LnVectorProb + log(sum(exp(LnVectorProb - scaling_factors),1));
 
    % sum all log values to get global model probability
    LnTotalProb = sum(LnVectorProb_new);
end




%--------------------------------------------------------------------------
% function for plotting a result of EM-estimation
% 
% Inputs:
%   model: Gaussian Mixture Model Parametes (structure)
%   features: feature vectors
% 
%  - plots a mixture model in a 3dim- plot (feature vector has to be 3-dimensional!)
%  - Feature vectors are plotted as green dots
%  - Means of components: red circles
%  - Covariance matrices: plotted using the three main axes of the
%    ellipsoid
function PlotGMM(model, trainVect)

    % number of components in the model
    n_comp = numel(model.weight);
    n_dims = size(trainVect, 2);
    
    hold on;
    % plot feature vector points
    plot3(trainVect(:,1),trainVect(:,2),trainVect(:,3), 'g.','MarkerSize',7);

    % plot elements of the estimated components:
    for i=1:n_comp
        
        % eigenvektor / eigenwert decomposition
        [eVec,eVal] = eig(squeeze(model.covar(i,:,:)));

        % plotting of mean values
        mean = squeeze(model.mean(i,:));
        plot3(mean(1),mean(2),mean(3),'ro');

        % derivation and plotting of the three main axes of the cvariance
        % matrices
        for i=1:n_dims
            devVec = (sqrt(eVal(i,i)) * eVec(:,i))*[-1,1];
            plot3(mean(1) + devVec(1,:), mean(2) + devVec(2,:), mean(3) + devVec(3,:),'b');
        end
    end

    % rotate 3D view and setting of title
    hold off;
    view([19,25]);
    grid('on');
    title(['Gaussian Mixture Model (',num2str(n_comp),' components)']);
end

% ------------------------------------------------------------------------
% function for splitting a single component and initialization 2 new ones
%
% Inputs:
%
%   model: GMM parameters
%   features: feature vectors
%   features(j,:): feature vector of a single pixel
%   newModel: updated model with new initialized component
%
% adds a new component (cluster) to the current model
%
% --> analyzes the current model and identifies the weakest component
% --> weak: doesn't fit well to the corresponding feature vectors
% the weakest component will be splitted into two new ones
function NewModel = InitNewComponent(model, trainVect)

    % number of components
    n_comp = numel(model.weight);

    % Number of dimensions (shall be three here!)
    n_dims = size(trainVect, 2);

    % the biggest component will be splitted to get a balanced size of
    % components --> not the optimal criterium!!!!
    % size corresponds to weights...
    [ignore, splitComp] = max(model.weight);

    % calculate new weight vector, mean and covariance
    newWeight = zeros(n_comp+1,1);
    newMean = zeros(n_comp+1,n_dims);
    newCovar = zeros(n_comp+1,n_dims,n_dims);

    % copy old values into new arrays
    newWeight(1:n_comp) = model.weight;
    newMean(1:n_comp,:) = model.mean;
    newCovar(1:n_comp,:,:) = model.covar;

    % Component splitComp will be splitted along its dominant axis
    [eVec,eVal] = eig(squeeze(newCovar(splitComp,:,:)));
    [ignore, majAxis] = max(diag(eVal));
    devVec = sqrt(eVal(majAxis,majAxis)) * eVec(:,majAxis)';

    % initialize new component
    % --> half of the points --> half weight
    newWeight(n_comp+1) = 0.5*newWeight(splitComp);
    % shift new mean to half of length along dominant axis
    newMean(n_comp+1,:) = newMean(splitComp,:) - 0.5*devVec;
    % make covariance a little bit smaller
    newCovar(n_comp+1,:,:) = newCovar(splitComp,:,:) / 4.0;

    % update also the (old) splitted component
    % also half of the points
    newWeight(splitComp) = newWeight(n_comp+1);
    % shift comonent center to other direction along dominant axis
    newMean(splitComp,:) = newMean(n_comp+1,:) + devVec;
    % take same smaller covariance matrix
    newCovar(splitComp,:,:) = newCovar(n_comp+1,:,:);

    % store new parameters in model
    NewModel.weight = newWeight;
    NewModel.mean = newMean
    NewModel.covar = newCovar;
end
