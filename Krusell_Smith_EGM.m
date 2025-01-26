%% Krusell-Smith Model - Endogenous Grid Method
clear all; close all; clc;

%% Model Parameters
beta = 0.99;       alpha = 0.36;      delta = 0.025;     
theta = 1;         k_min = 0.0001;    k_max = 1000;     
k_size = 100;      K_min = 30;        K_max = 50;       
K_size = 4;        z_grid = [1.01, 0.99];  eps_grid = [1, 0];
ug = 0.04;         ub = 0.10;         mu = 0;         
l_bar = 1/(1-ub);  T = 1100;          population = 10000;
T_discard = 100;   max_iter_B = 100;   tol_B = 1e-6;     
update_B = 0.3;    max_egm = 10000;      tol_egm = 1e-6;

%% Grid Construction
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
B = [0, 1, 0, 1];
K_ts = zeros(T,1); 
k_population = ones(population,1)*K_grid(1);

%% Precompute Wage/Interest Tables
w_table = zeros(s_size, K_size);
r_table = zeros(s_size, K_size);
for s_i = 1:s_size
    z = s_grid(s_i,1); eps_val = s_grid(s_i,2);
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

%% Main ALM Loop
for B_iter = 1:max_iter_B
    fprintf('\n--- Capital Law of Motion Iteration %d/%d ---\n', B_iter, max_iter_B);
    
    %% EGM Policy Iteration
    egm_converged = false;
    for egm_iter = 1:max_egm
        k_opt_old = k_opt;
        
        for s_i = 1:s_size
            z = s_grid(s_i,1); eps = s_grid(s_i,2);
            
            for K_i = 1:K_size
                K = K_grid(K_i);
                
                % Compute K' using Capital Law of Motion
                if z == z_grid(1)
                    logK_prime = B(1) + B(2)*log(K);
                else
                    logK_prime = B(3) + B(4)*log(K);
                end
                K_prime = exp(logK_prime);
                [~, K_prime_idx] = min(abs(K_grid - K_prime));
                
                % Current wage and interest rate
                L = l_bar * (1 - ug*(z==z_grid(1)) - ub*(z==z_grid(2)));
                r = r_table(s_i, K_i);
                w = w_table(s_i, K_i);
                
                % EGM Step
                k_current_vec = zeros(k_size,1);
                for kp_i = 1:k_size
                    kp = k_grid(kp_i);
                    expected_marginal = 0;
                    
                    for s_j = 1:s_size
                        prob = P(s_i, s_j);
                        z_next = s_grid(s_j,1); eps_next = s_grid(s_j,2);
                        
                        % Compute K'' using Capital Law of Motion for next period
                        if z_next == z_grid(1)
                            logK_dprime = B(1) + B(2)*log(K_prime);
                        else
                            logK_dprime = B(3) + B(4)*log(K_prime);
                        end
                        K_dprime = exp(logK_dprime);
                        [~, K_dprime_idx] = min(abs(K_grid - K_dprime));
                        
                        % Next period rates
                        L_next = l_bar * (1 - ug*(z_next==z_grid(1)) - ub*(z_next==z_grid(2)));
                        r_next = alpha*z_next*K_dprime^(alpha-1)*L_next^(1-alpha);
                        w_next = (1-alpha)*z_next*K_dprime^alpha*L_next^(-alpha);
                        
                        % Next period resources and consumption
                        resources_next = (1 + r_next - delta)*kp + w_next*eps_next*l_bar;
                        kp_next = interp1(k_grid, k_opt(:, K_dprime_idx, s_j), kp, 'pchip');
                        c_next = resources_next - kp_next;
                        c_next = max(c_next, 1e-8);
                        
                        expected_marginal = expected_marginal + prob*(1 + r_next - delta)/c_next;
                    end
                    
                    % Current consumption and capital
                    c = 1/(beta*expected_marginal);
                    k_current = (c + kp - w*eps*l_bar)/(1 + r - delta);
                    k_current_vec(kp_i) = k_current;
                end
                
                % Interpolate policy function
                [k_sorted, sort_idx] = sort(k_current_vec);
                kp_sorted = k_grid(sort_idx);
                valid = k_sorted >= k_min & k_sorted <= k_max;
                F = griddedInterpolant(k_sorted(valid), kp_sorted(valid), 'pchip', 'nearest');
                k_opt_new = F(k_grid);
                k_opt_new = max(min(k_opt_new, k_max), k_min);
                k_opt(:,K_i,s_i) = k_opt_new;
            end
        end
        
        % Check convergence
        diff = max(abs(k_opt(:) - k_opt_old(:)));
        fprintf('EGM Iter %d, Diff: %.2e\n', egm_iter, diff);
        if diff < tol_egm
            egm_converged = true; break; 
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