%% Krusell-Smith Model 
clear all; close all; clc;

%% Model Parameters
beta = 0.99;       alpha = 0.36;      delta = 0.025;     
theta = 1;         k_min = 0.0001;    k_max = 1000;     
k_size = 100;      K_min = 30;        K_max = 50;       
K_size = 4;        z_grid = [1.01, 0.99];  eps_grid = [1, 0];
ug = 0.04;         ub = 0.10;         mu = 0;         
l_bar = 1/(1-ub);  T = 1100;          population = 10000;
T_discard = 100;   max_iter_B = 100;   tol_B = 1e-6;     
update_B = 0.3;    max_vfi = 10000;    tol_vfi = 1e-6;
howard_steps = 50;      

%% Accelerated Grid Construction
k_grid = linspace(0, 1, k_size).^7 * (k_max - k_min) + k_min;
k_grid(1) = k_min; k_grid(end) = k_max;
K_grid = linspace(K_min, K_max, K_size);
[Z, Eps] = meshgrid(z_grid, eps_grid);
s_grid = [Z(:), Eps(:)]; 
s_size = size(s_grid, 1);

%% Transition Matrix Construction 
zg_ave_dur = 8; zb_ave_dur = 8;
pgg = 1 - 1/zg_ave_dur; pbb = 1 - 1/zb_ave_dur;
pgb = 1 - pgg; pbg = 1 - pbb;

ug_ave_dur = 1.5; ub_ave_dur = 2.5;
puu_rel_gb2bb = 1.25; puu_rel_bg2gg = 0.75;

p00_gg = 1 - 1/ug_ave_dur;
p00_bb = 1 - 1/ub_ave_dur;
p00_gb = puu_rel_gb2bb * p00_bb;
p00_bg = puu_rel_bg2gg * p00_gg;

p01_gg = 1 - p00_gg; p01_bb = 1 - p00_bb;
p01_gb = 1 - p00_gb; p01_bg = 1 - p00_bg;

p10_gg = (ug - ug*p00_gg)/(1 - ug);
p10_bb = (ub - ub*p00_bb)/(1 - ub);
p10_gb = (ub - ug*p00_gb)/(1 - ug);
p10_bg = (ug - ub*p00_bg)/(1 - ub);

p11_gg = 1 - p10_gg; p11_bb = 1 - p10_bb;
p11_gb = 1 - p10_gb; p11_bg = 1 - p10_bg;

P = zeros(4,4);
P(1,1) = pgg*p11_gg; P(1,2) = pgb*p11_gb;
P(1,3) = pgg*p10_gg; P(1,4) = pgb*p10_gb;
P(2,1) = pbg*p11_bg; P(2,2) = pbb*p11_bb;
P(2,3) = pbg*p10_bg; P(2,4) = pbb*p10_bb;
P(3,1) = pgg*p01_gg; P(3,2) = pgb*p01_gb;
P(3,3) = pgg*p00_gg; P(3,4) = pgb*p00_gb;
P(4,1) = pbg*p01_bg; P(4,2) = pbb*p01_bb;
P(4,3) = pbg*p00_bg; P(4,4) = pbb*p00_bb;

%% Shock Simulation
fprintf('Simulating shocks...\n');
zi_shock = zeros(T,1);  % Aggregate shocks
zi_shock(1) = 1;  % Start in good state (z=1.01)
for t = 2:T
    if zi_shock(t-1) == 1
        zi_shock(t) = 1 + (rand > pgg);
    else
        zi_shock(t) = 1 + (rand > pbb);
    end
end
zi_shock = zi_shock - 1; 

epsi_shock = zeros(T, population);  % Idiosyncratic shocks
epsi_shock(1,:) = (rand(1,population) > ug) + 1;
for t = 2:T
    for i = 1:population
        current_z = zi_shock(t);
        prev_z = zi_shock(t-1);
        prev_eps = epsi_shock(t-1,i);
        
        if current_z == 0 && prev_z == 0
            Peps = [p11_gg, p10_gg; p01_gg, p00_gg];
        elseif current_z == 0 && prev_z == 1
            Peps = [p11_bg, p10_bg; p01_bg, p00_bg];
        elseif current_z == 1 && prev_z == 0
            Peps = [p11_gb, p10_gb; p01_gb, p00_gb];
        else
            Peps = [p11_bb, p10_bb; p01_bb, p00_bb];
        end
        
        if prev_eps == 1
            epsi_shock(t,i) = 1 + (rand > Peps(1,1));
        else
            epsi_shock(t,i) = 1 + (rand > Peps(2,1));
        end
    end
end

%% Initialize Arrays
k_opt = 0.9 * repmat(k_grid', [1, K_size, s_size]);
value = log(0.1/0.9 * k_opt) / (1 - beta);
B = [0, 1, 0, 1];
K_ts = zeros(T,1); 
k_population = ones(population,1)*K_grid(1);

%% Precompute Wage/Interest Tables
fprintf('Precomputing wage/interest tables...\n');
w_table = zeros(s_size, K_size);
r_table = zeros(s_size, K_size);
for s_i = 1:s_size
    z = s_grid(s_i,1);
    eps_val = s_grid(s_i,2);
    for K_i = 1:K_size
        K = K_grid(K_i);
        L = l_bar * (1 - ug*(z==z_grid(1)) - ub*(z==z_grid(2)));
        w_table(s_i,K_i) = (1-alpha)*z*K^alpha*L^(-alpha);
        r_table(s_i,K_i) = alpha*z*K^(alpha-1)*L^(1-alpha);
    end
end

%% State Lookup Table
fprintf('Precomputing state lookup table...\n');
state_lookup = containers.Map('KeyType','char','ValueType','double');
for s_i = 1:s_size
    z_norm = round(s_grid(s_i,1), 6);
    eps_norm = round(s_grid(s_i,2), 6);
    key = sprintf('%.6f,%.6f', z_norm, eps_norm);
    state_lookup(key) = s_i;
end

%% Value Interpolants
fprintf('Precomputing value interpolants...\n');
V_interp = cell(K_size, s_size);
for Kp_i = 1:K_size
    for s_j = 1:s_size
        V_interp{Kp_i,s_j} = griddedInterpolant(k_grid, value(:,Kp_i,s_j), 'pchip');
    end
end

%% Main Optimization Loop
for B_iter = 1:max_iter_B
    fprintf('\n--- Capital Law of Motion Iteration %d/%d ---\n', B_iter, max_iter_B);
    
    %% Howard-Accelerated VFI
    vfi_converged = false;
    for vfi_iter = 1:max_vfi
        tic;
        value_old = value;
        
        % Policy Improvement (Every 5 iterations)
        if mod(vfi_iter-1, 5) == 0
            resources = zeros(k_size, K_size, s_size);
            for s_i = 1:s_size
                for K_i = 1:K_size
                    resources(:,K_i,s_i) = (r_table(s_i,K_i) + 1 - delta)*k_grid + ...
                        w_table(s_i,K_i)*(s_grid(s_i,2)*l_bar + (1-s_grid(s_i,2))*mu);
                end
            end
            
            for s_i = 1:s_size
                for K_i = 1:K_size
                    kp_max = min(resources(:,K_i,s_i), k_max);
                    for k_i = 1:k_size
                        f = @(kp) -bellman_value(kp, k_grid(k_i), K_grid(K_i), s_i, ...
                                               V_interp, B, P, beta, alpha, delta, ...
                                               theta, z_grid, K_grid, ug, ub, s_grid, l_bar);
                        [kp_opt, fval] = fminbnd(f, k_min, kp_max(k_i));
                        k_opt(k_i,K_i,s_i) = kp_opt;
                    end
                end
            end
        end
        
        %% Modified Howard Evaluation
        for howard = 1:howard_steps
            value_new = value;
            for s_i = 1:s_size
                for K_i = 1:K_size
                    for k_i = 1:k_size
                        val = bellman_value(k_opt(k_i,K_i,s_i),...
                            k_grid(k_i), K_grid(K_i), s_i, V_interp, B, P, beta,...
                            alpha, delta, theta, z_grid, K_grid, ug, ub, s_grid, l_bar);
                        value_new(k_i,K_i,s_i) = val;
                    end
                end
            end
            value = value_new;
            
            % Update interpolants
            for Kp_i = 1:K_size
                for s_j = 1:s_size
                    V_interp{Kp_i,s_j}.Values = value(:,Kp_i,s_j);
                end
            end
        end
        
        % Convergence check
        diff = max(abs(value(:) - value_old(:))./(abs(value_old(:)) + 1e-10));
        elapsed = toc;
        fprintf('VFI Iter %d/%d, RelDiff: %.2e, Time: %.1fs\n', ...
                vfi_iter, max_vfi, diff, elapsed);
        
        if diff < tol_vfi
            vfi_converged = true; 
            break; 
        end
    end
    
    %% Capital Path Simulation
    fprintf('Simulating capital path...\n');
    K_ts(1) = mean(k_population);
    
    state_indices = containers.Map('KeyType','char','ValueType','double');
    all_eps = unique(epsi_shock);
    for e = all_eps'
        eps_val = eps_grid(e);
        for z_val = z_grid
            key = sprintf('%.6f,%.6f', round(z_val,6), round(eps_val,6));
            if isKey(state_lookup, key)
                state_indices(key) = state_lookup(key);
            end
        end
    end

    for t = 1:T-1
        if mod(t,100) == 0
            fprintf('Progress: %d/%d (%.1f%%)\n', t, T-1, 100*t/(T-1));
        end
        
        z = z_grid(zi_shock(t)+1);
        eps_vals = eps_grid(epsi_shock(t,:));
        keys = arrayfun(@(e) sprintf('%.6f,%.6f', round(z,6), round(e,6)), eps_vals, 'UniformOutput', false);
        
        current_states = cell2mat(values(state_indices, keys));
        current_states = current_states(:);
        
        new_k = zeros(population, 1);
        for s = unique(current_states)'
            mask = (current_states == s);
            mask = mask(:);
            
            if sum(mask) == 0, continue; end
            
            F = griddedInterpolant({k_grid, K_grid}, k_opt(:,:,s));
            k_points = k_population(mask);
            K_points = K_ts(t) * ones(numel(k_points), 1);
            new_k(mask) = F(k_points(:), K_points(:));
        end
        k_population = new_k;
        K_ts(t+1) = mean(k_population);
    end
    
    %% Capital Law of Motion Update
    fprintf('Updating Capital Law of Motion coefficients...\n');
    X_g = []; Y_g = []; X_b = []; Y_b = [];
    for t = T_discard:T-1
        if zi_shock(t) == 0
            X_g = [X_g; 1, log(K_ts(t))];
            Y_g = [Y_g; log(K_ts(t+1))];
        else
            X_b = [X_b; 1, log(K_ts(t))];
            Y_b = [Y_b; log(K_ts(t+1))];
        end
    end
    
    B_new = zeros(4,1);
    R2_g = 0; R2_b = 0;  % Initialize R² values
    
    % Good state regression
    if ~isempty(X_g)
        B_new(1:2) = X_g \ Y_g;
        Y_pred_g = X_g*B_new(1:2);
        resid_g = Y_g - Y_pred_g;
        SS_res_g = sum(resid_g.^2);
        SS_tot_g = sum((Y_g - mean(Y_g)).^2);
        R2_g = 1 - SS_res_g/SS_tot_g;
    end
    
    % Bad state regression
    if ~isempty(X_b)
        B_new(3:4) = X_b \ Y_b;
        Y_pred_b = X_b*B_new(3:4);
        resid_b = Y_b - Y_pred_b;
        SS_res_b = sum(resid_b.^2);
        SS_tot_b = sum((Y_b - mean(Y_b)).^2);
        R2_b = 1 - SS_res_b/SS_tot_b;
    end
    
    diff_B = max(abs(B_new' - B));
    fprintf(['Capital Law of Motion Update: B_new = [%.4f, %.4f, %.4f, %.4f], '...
             'Diff: %.2e\nR²: Good=%.4f, Bad=%.4f\n'], ...
             B_new(1), B_new(2), B_new(3), B_new(4), diff_B, R2_g, R2_b);
    
    if diff_B < tol_B
        fprintf('Capital Law of Motion converged after %d iterations!\n', B_iter);
        break; 
    end
    B = update_B*B_new' + (1-update_B)*B;
end

%% Plot Results
% Preallocate K_ts_approx with the same size as K_ts
K_ts_approx = zeros(size(K_ts));

% Initialize the approximate ALM with the initial value
K_ts_approx(T_discard) = K_ts(T_discard);

% Compute the approximate ALM for capital
for t = T_discard:(length(zi_shock) - 1)
    K_ts_approx(t + 1) = compute_approxKprime(K_ts_approx(t), zi_shock(t), B);
end

figure;
subplot(2,1,1);
plot(K_ts(T_discard+1:end), '-r');
hold on;
plot(K_ts_approx(T_discard+1:end), '--b');
title('Aggregate Capital Law of Motion');
xlabel('Time'); ylabel('K'); legend('True', 'Approximation')

subplot(2,1,2);
K_lim = linspace(min(K_ts), max(K_ts), 100);
plot(K_lim, exp(B(1)+B(2)*log(K_lim)), 'b-',...
     K_lim, exp(B(3)+B(4)*log(K_lim)), 'r-',...
     K_lim, K_lim, 'k--');
title('Tomorrow vs Today Aggregate Capital');
legend('Good State', 'Bad State', '45° Line');
xlabel('K_t'); ylabel('K_{t+1}');

%% Functions
% Function Bellman
function utility = bellman_value(kp, k, K, s_i, V_interp, B, P, beta, alpha,...
                                delta, theta, z_grid, K_grid, ug, ub, s_grid, l_bar)
    % Current state
    current_z = z_grid((s_i <= 2) + 1);
    
    % Next period's K prediction with bounds
    if current_z == z_grid(1)
        Kp = exp(B(1) + B(2)*log(max(K, 1e-8)));
    else
        Kp = exp(B(3) + B(4)*log(max(K, 1e-8)));
    end
    Kp = max(min(Kp, K_grid(end)), K_grid(1));
    
    % Find nearest K grid index
    [~, Kp_idx] = min(abs(K_grid - Kp));
    
    % Expected value calculation with interpolation bounds
    expec = 0;
    for s_next = 1:size(P,2)
        expec = expec + P(s_i, s_next) * V_interp{Kp_idx,s_next}(max(min(kp, V_interp{Kp_idx,s_next}.GridVectors{1}(end)), V_interp{Kp_idx,s_next}.GridVectors{1}(1)));
    end
    
    % Consumption calculation
    L = l_bar * (1 - ug*(current_z==z_grid(1)) - ub*(current_z==z_grid(2)));
    r_val = alpha*current_z*K^(alpha-1)*L^(1-alpha);
    w_val = (1-alpha)*current_z*K^alpha*L^(-alpha);
    c = (r_val + 1 - delta)*k + w_val*(s_grid(s_i,2)*l_bar) - kp;
    
    % Utility calculation
    c = max(c, 1e-10);
    if theta == 1
        utility = log(c) + beta * expec;
    else
        utility = (c^(1-theta))/(1-theta) + beta * expec;
    end
end

% Function to compute approximate K prime
function K_prime = compute_approxKprime(K, z, B)
    if z == 0
        K_prime = exp(B(1) + B(2) * log(K));
    elseif z == 1
        K_prime = exp(B(3) + B(4) * log(K));
    else
        error('Unexpected value of z.');
    end
end