clearvars;
clc;
close all;

%% Parameterization
beta = 0.96;         % Discount factor
sigma = 5;           % Risk aversion
alpha = 0.36;        % Capital share
delta = 0.08;        % Depreciation rate
rho = 0.6;           % Autocorrelation of productivity
b = 0;               % Debt limit
sigma_e = 0.2;       % Standard deviation of productivity shocks
N = 7;               % Number of states in Markov process
psi = 1;             % Labor disutility parameter
eta = 2;             % Labor elasticity

%% Tauchen method for Markov process
l_grid = zeros(1,N);
for i = 1:N
    l_grid(i) = (i - 4) * sigma_e;
end

intervals = [-inf, -2.5 * sigma_e, -1.5 * sigma_e, -0.5 * sigma_e, 0.5 * sigma_e, 1.5 * sigma_e, 2.5 * sigma_e, inf];
P = zeros(N, N);
sd = sigma_e * sqrt(1 - rho^2);
for i = 1:N
    for j = 1:N
        P(i, j) = integral(@(x) normpdf(x, rho * l_grid(i), sd), intervals(j), intervals(j + 1));
    end
end

%% Find stationary distribution
Amatrix = [P' - eye(N); ones(1, N)];
Bmatrix = [zeros(N, 1); 1];

stationary_dist = Amatrix \ Bmatrix;
s = exp(l_grid);
labor = s * stationary_dist;
L = s/labor;

%% Value Function Iteration
max_iter = 1000;
tol = 1e-5;
grid_size = 400;

% Asset grid boundaries
wmin = (1-alpha)*(alpha/((1/beta - 1) + delta))^(alpha/(1-alpha));
amin = min(b, wmin*s(1));
kmax = delta^(1/((alpha-1)));
amax = kmax^alpha + (1-delta)*kmax;

a_grid = amin + (amax - amin) * (linspace(0, 1, grid_size).^2);

v_old = zeros(N, grid_size);  % Initial guess for value function
policy_k = zeros(N, grid_size);  % Policy for capital
policy_c = zeros(N, grid_size);  % Policy for consumption
policy_l = zeros(N, grid_size);  % Policy for labor

r_guess = 0.04;  % Initial guess for interest rate

% Precompute labor choices
labor_choice = linspace(0.01, 1.5, 10);  % Avoid zero labor to prevent log(0)

for iter = 1:max_iter
    r = r_guess;
    w = (1 - alpha) * (alpha / (r + delta))^(alpha / (1 - alpha));
    
    % Precompute expected value EV for future states
    EV = beta * P * v_old;  % This handles the expected future value for all states
    
    % Vectorize the outer loop over productivity states and asset levels
    for i = 1:N
        current_s = s(i);  % Productivity for state i
        
        % Compute current capital and wage
        for j = 1:grid_size
            current_a = a_grid(j);  % Current asset level

            % Compute consumption and utility for all labor choices in one step
            [K_next, L_grid] = meshgrid(a_grid, labor_choice);  % Create a grid of future capital and labor choices
            c_labor = (1 + r) * current_a + w * current_s * L_grid - K_next;  % Future consumption for each labor and capital
            
            valid_c = c_labor > 0;  % Ensure consumption is positive
            
            if any(valid_c(:))
                % Calculate utility where valid consumption
                utility = -inf(size(c_labor));  % Initialize utility
                valid_indices = find(valid_c);  % Find valid consumption indices
                
                % Extract valid values for both consumption and labor
                valid_c_labor = c_labor(valid_c);
                valid_L_supply = L_grid(valid_c);
                
                % Compute the utility for valid choices
                utility(valid_c) = (valid_c_labor.^(1 - sigma) - 1) / (1 - sigma) ...
                    - psi * (valid_L_supply.^(1 + eta)) / (1 + eta);  % Element-wise labor disutility
                
                % Total value function: utility + discounted expected value (vectorized)
                total_value = utility + EV(i, :);  % Combine with future expected value
                
                % Maximize total value over labor and future capital choices
                [max_value, idx] = max(total_value(:));  % Find the maximum value
                [opt_l_idx, opt_k_idx] = ind2sub(size(total_value), idx);  % Get indices for optimal labor and capital
                
                % Update policy functions
                policy_l(i, j) = labor_choice(opt_l_idx);
                policy_k(i, j) = a_grid(opt_k_idx);
                policy_c(i, j) = c_labor(opt_l_idx, opt_k_idx);
                v_new(i, j) = max_value;
            end
        end
    end
    
    % Check for convergence
    if max(abs(v_new - v_old), [], 'all') < tol
        disp(['Converged in ', num2str(iter), ' iterations']);
        break;
    end
    
    v_old = v_new;  % Update value function for next iteration
    disp(iter);
end

disp('Value function iteration complete');

%% Simulation with Piecewise Linear Interpolation

num_simulations = 10000;
sim_z = zeros(num_simulations, 1);
sim_k = zeros(num_simulations, 1);
sim_c = zeros(num_simulations, 1);
sim_l = zeros(num_simulations, 1);  % Labor supply
sim_y = zeros(num_simulations, 1);  % Net income
sim_s = zeros(num_simulations, 1);  % Savings
sim_z(1) = randi(N);
sim_k(1) = a_grid(randi(grid_size));

for t = 2:num_simulations
    sim_z(t) = find(rand < cumsum(P(sim_z(t - 1), :)), 1);
    k_prev = sim_k(t - 1);
    k_policy = policy_k(sim_z(t), :);  % Policy function for current state
    sim_k(t) = interp1(a_grid, k_policy, k_prev, 'linear', 'extrap');
    sim_l(t) = interp1(a_grid, policy_l(sim_z(t), :), k_prev, 'linear', 'extrap');  % Labor supply
    sim_c(t) = interp1(a_grid, policy_c(sim_z(t), :), k_prev, 'linear', 'extrap');

    sim_y(t) = r * sim_k(t) + w * s(sim_z(t)) * sim_l(t);
    sim_s(t) = r * sim_k(t) + w * s(sim_z(t)) * sim_l(t) + delta * sim_k(t) - sim_c(t);
end

sim_k = sim_k(:);
sim_z = sim_z(:);

agg_k_supply = mean(sim_k(:)); 

%% Iteratively update r until capital demand equals capital supply

max_r_iter = 10;
r_tol = 1e-5;
r_low = -0.05;
r_high = 1/beta - 1;
r_mid = (r_low + r_high) / 2;
k_demand = zeros(max_r_iter, 1);
k_supply = zeros(max_r_iter, 1);
r_history = zeros(max_r_iter, 1);

for r_iter = 1:max_r_iter
    r_mid = (r_low + r_high) / 2;
    r_guess = r_mid;
    
    % Recalculate Value Function Iteration with updated r
    for iter = 1:max_iter
    r = r_guess;
    w = (1 - alpha) * (alpha / (r + delta))^(alpha / (1 - alpha));
    
    % Precompute expected value EV for future states
    EV = beta * P * v_old;  % This handles the expected future value for all states
    
    % Vectorize the outer loop over productivity states and asset levels
    for i = 1:N
        current_s = s(i);  % Productivity for state i
        
        % Compute current capital and wage
        for j = 1:grid_size
            current_a = a_grid(j);  % Current asset level

            % Compute consumption and utility for all labor choices in one step
            [K_next, L_grid] = meshgrid(a_grid, labor_choice);  % Create a grid of future capital and labor choices
            c_labor = (1 + r) * current_a + w * current_s * L_grid - K_next;  % Future consumption for each labor and capital
            
            valid_c = c_labor > 0;  % Ensure consumption is positive
            
            if any(valid_c(:))
                % Calculate utility where valid consumption
                utility = -inf(size(c_labor));  % Initialize utility
                valid_indices = find(valid_c);  % Find valid consumption indices
                
                % Extract valid values for both consumption and labor
                valid_c_labor = c_labor(valid_c);
                valid_L_supply = L_grid(valid_c);
                
                % Compute the utility for valid choices
                utility(valid_c) = (valid_c_labor.^(1 - sigma) - 1) / (1 - sigma) ...
                    - psi * (valid_L_supply.^(1 + eta)) / (1 + eta);  % Element-wise labor disutility
                
                % Total value function: utility + discounted expected value (vectorized)
                total_value = utility + EV(i, :);  % Combine with future expected value
                
                % Maximize total value over labor and future capital choices
                [max_value, idx] = max(total_value(:));  % Find the maximum value
                [opt_l_idx, opt_k_idx] = ind2sub(size(total_value), idx);  % Get indices for optimal labor and capital
                
                % Update policy functions
                policy_l(i, j) = labor_choice(opt_l_idx);
                policy_k(i, j) = a_grid(opt_k_idx);
                policy_c(i, j) = c_labor(opt_l_idx, opt_k_idx);
                v_new(i, j) = max_value;
            end
        end
    end
    
    % Check for convergence
    if max(abs(v_new - v_old), [], 'all') < tol
        disp(['Converged in ', num2str(iter), ' iterations']);
        break;
    end
    
    v_old = v_new;  % Update value function for next iteration
   end
    
    % Simulate again with updated policy function
    for t = 2:num_simulations
        sim_z(t) = find(rand < cumsum(P(sim_z(t - 1), :)), 1);
        k_prev = sim_k(t - 1);
        k_policy = policy_k(sim_z(t), :);
        sim_k(t) = interp1(a_grid, k_policy, k_prev, 'linear', 'extrap');
        sim_l(t) = interp1(a_grid, policy_l(sim_z(t), :), k_prev, 'linear', 'extrap');  % Labor supply
        sim_c(t) = interp1(a_grid, policy_c(sim_z(t), :), k_prev, 'linear', 'extrap');

        sim_y(t) = r * sim_k(t) + w * s(sim_z(t)) * sim_l(t);
        sim_s(t) = r * sim_k(t) + w * s(sim_z(t)) * sim_l(t) + delta * sim_k(t) - sim_c(t);
    end
    
    agg_k_supply = mean(sim_k(:));
    k_supply(r_iter) = agg_k_supply;
    k_demand(r_iter) = labor * (alpha / (r_guess + delta))^(1 / (1 - alpha));
    r_history(r_iter) = r;
    
    if abs(agg_k_supply - k_demand(r_iter)) < r_tol
        break
    elseif agg_k_supply > k_demand(r_iter)
        r_high = r_guess;
    else
        r_low = r_guess;
    end
    disp(r_iter)
end

savings_percent = (delta * alpha / (r + delta)) * 100;
r_percent = r * 100;

r_history = sort(r_history, "ascend");
k_demand = sort(k_demand, "descend");
k_supply = sort(k_supply, "ascend");

%% Plotting

figure(1);
plot(k_demand, r_history, 'r', 'LineWidth', 2);
xlim([0, 18]);
hold on;
plot(k_supply, r_history, 'b--', 'LineWidth', 2);
xlim([0, 18]);
yline((1 - beta) / beta);
xlim([0, 18]);
xlabel('Total Assets');
ylabel('Interest Rate');
legend('Capital Demand', 'Capital Supply');
title('Steady State: mu=5, rho=0.6, sig=0.2');
grid on;

figure(2);
plot(a_grid, policy_k(1, :), 'r:', 'LineWidth', 2);
xlim([0, 20]);
ylim([0, 20]);
hold on;
plot(a_grid, policy_k(2, :), 'b--', 'LineWidth', 2);
xlim([0, 20]);
ylim([0, 20]);
xlabel('Total Resources');
ylabel('Asset Holding');
legend('Lmin', 'Lmax');
title('Asset Demand Functions: mu=5, rho=0.6, sig=0.2');
grid on;

%% Densities

[f_w, xi_w] = ksdensity(sim_k);
[f_c, xi_c] = ksdensity(sim_c);
[f_y, xi_y] = ksdensity(sim_y);
[f_s, xi_s] = ksdensity(sim_s);

% Normalize the density estimates
f_w_normalized = f_w / (sum(f_w)); 
f_c_normalized = f_c / (sum(f_c)); 
f_y_normalized = f_y / (sum(f_y));
f_s_normalized = f_s / (sum(f_s));

figure;
plot(xi_w, f_w_normalized, '-b', 'LineWidth', 2);
xlim([0,16]);
title('Density of Wealth');
xlabel('Value');
ylabel('Density');
grid on;

figure;
plot(xi_c, f_c_normalized, 'LineWidth', 2);
hold on;
plot(xi_y, f_y_normalized, 'LineWidth', 2);
plot(xi_s, f_s_normalized, 'LineWidth', 2);
% hold off;    
title('Density Consumption, Net Income, and Savings');
xlabel('Value');
ylabel('Density');
legend('Consumption', 'Net Income', 'Savings');
grid on;

%% Histograms

figure;
histogram(sim_k, 'Normalization', 'probability');
xlim([0,16]);
title('Histogram of Wealth');
xlabel('Values');
ylabel('Probability');

figure;
histogram(sim_c, 'Normalization', 'probability');
title('Histogram of Consumption');
xlabel('Value');
ylabel('Probability');

figure;
histogram(sim_y, 'Normalization', 'probability');
title('Histogram of Net Income');
xlabel('Value');
ylabel('Probability');

figure;
histogram(sim_s, 'Normalization', 'probability');
title('Histogram of Savings');
xlabel('Value');
ylabel('Probability');

%% Lorenz Curves

% Sort the data
sorted_k = sort(sim_k);
sorted_c = sort(sim_c);
sorted_y = sort(sim_y);
sorted_s = sort(sim_s);

% Compute the cumulative sum and the cumulative sum of population share
cumulative_k = cumsum(sorted_k) / sum(sorted_k);
cumulative_population_k = (1:length(sorted_k)) / length(sorted_k);

cumulative_c = cumsum(sorted_c) / sum(sorted_c);
cumulative_population_c = (1:length(sorted_c)) / length(sorted_c);

cumulative_y = cumsum(sorted_y) / sum(sorted_y);
cumulative_population_y = (1:length(sorted_y)) / length(sorted_y);

cumulative_s = cumsum(sorted_s) / sum(sorted_s);
cumulative_population_s = (1:length(sorted_s)) / length(sorted_s);

% Calculate the area under the Lorenz curve
lorenz_area_k = trapz(cumulative_population_k, cumulative_k);
lorenz_area_c = trapz(cumulative_population_c, cumulative_c);
lorenz_area_y = trapz(cumulative_population_y, cumulative_y);
lorenz_area_s = trapz(cumulative_population_s, cumulative_s);

% Calculate the Gini coefficient
gini_coefficient_k = 1 - 2 * lorenz_area_k;
gini_coefficient_c = 1 - 2 * lorenz_area_c;
gini_coefficient_y = 1 - 2 * lorenz_area_y;
gini_coefficient_s = 1 - 2 * lorenz_area_s;

disp(['Gini Coefficient for Wealth: ', num2str(gini_coefficient_k)]);
disp(['Gini Coefficient for Consumption: ', num2str(gini_coefficient_c)]);
disp(['Gini Coefficient for Net Income: ', num2str(gini_coefficient_y)]);
disp(['Gini Coefficient for Savings: ', num2str(gini_coefficient_s)]);

% Plot the Lorenz Curve
figure;
plot(cumulative_population_k, cumulative_k, "b-", 'LineWidth', 2);
hold on;
plot(cumulative_population_c, cumulative_c, "r-", 'LineWidth', 2);
plot(cumulative_population_y, cumulative_y, "y-", 'LineWidth', 2);
plot(cumulative_population_s, cumulative_s, "g-", 'LineWidth', 2);
plot([0 1], [0 1], 'k--'); % Line of equality
xlabel('Cumulative Share of Population');
ylabel('Cumulative Share of Wealth');
title('Lorenz Curves');
legend('Wealth', 'Consumption', 'Net Income', 'Savings');
grid on;