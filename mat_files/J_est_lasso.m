function [] = J_est_lasso()

files = dir('*.mat');

for file = files'
    filename = file.name;
    disp('================================================');
    disp(filename);
    
    data = load(filename);
    S = data.S;
    J = data.J;
    S(S==-1) = 0;

    indices_ = strfind(filename, '_');
    last_index = indices_(size(indices_,2));
    index_dot = strfind(filename, '.');
    likelyhood = filename(last_index+1:index_dot-1);

    if strcmp(likelyhood, 'logistic')
        likelyhood = 'logit';
    end

    num_neurons = size(S, 2);
    time_steps = size(S,1);
    J_est_1 = zeros(num_neurons, num_neurons);
    J_est_2 = zeros(num_neurons, num_neurons);

    disp(num_neurons);
    disp(time_steps);
    disp(likelyhood);
    
    x_i = S(1:time_steps-1, :);
    for i = 1:num_neurons
        y_i = S(2:time_steps, i);

        [B, info] = lassoglm(x_i, y_i, 'binomial', 'Link', likelyhood, 'CV', 5);

        J_est_1(i,:) = B(:, info.IndexMinDeviance);
        J_est_2(i,:) = B(:, info.Index1SE);
        disp(i);
    end

    S(S==0)=-1;

    new_file_name = strcat(filename(1:index_dot-1), '_J_est.mat');

    save(new_file_name, 'S', 'J', 'J_est_1', 'J_est_2');

end
end