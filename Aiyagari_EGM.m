clearvars;
clc;
close all;

%% Parameterization

beta = 0.96;         % Discount factor
sigma = 5;           % Risk aversion
alpha = 0.36;        % Capital share
delta = 0.08;        % Depreciation rate
b = 0;               % Debt limit
rho = 0.75;          % Autocorrelation of productivity
sigma_e = 0.75;      % Standard deviation of productivity shocks
N = 7;               % Number of states in Markov process
grid_size = 400;     % Number of asset grid points
tol = 1e-5;          % Convergence tolerance
max_iter = 1000;     % Maximum iterations for EGM

%% Tauchen method for Markov process

l_grid = zeros(1, N);
for i = 1:N
    l_grid(i) = (i - 4) * sigma_e;
end

intervals = [-inf, -2.5 * sigma_e, -1.5 * sigma_e, -0.5 * sigma_e, 0.5 * sigma_e, 1.5 * sigma_e, 2.5 * sigma_e, inf];

P = zeros(N, N);
sd = sigma_e * sqrt(1 - rho^2);
for i = 1:N
    for j = 1:N
        lower_bound = intervals(j);
        upper_bound = intervals(j + 1);
        % Numerical integration
        integrand = @(x) normpdf(x, rho * l_grid(i), sd);
        P(i, j) = integral(integrand, lower_bound, upper_bound);
    end
end

%% Stationary distribution and normalize labor

Amatrix = [P' - eye(N); ones(1, N)];
Bmatrix = [zeros(N, 1); 1];

stationary_dist = Amatrix \ Bmatrix;
s = exp(l_grid);
labor = s * stationary_dist;
L = s / labor;

%% EGM Method for Policy Iteration

wmin = (1-alpha)*(alpha/((1/beta - 1) + delta))^(alpha/(1-alpha));
amin = min(b, wmin*s(1)); 
kmax = delta^(1 / (alpha - 1));
amax = kmax^alpha + (1 - delta) * kmax;
a_grid = amin + (amax - amin) * (linspace(0, 1, grid_size).^2)';

r = 0.04; % Initial guess for interest rate

% Compute the wage rate given r
w = (1 - alpha) * (alpha / (r + delta))^(alpha / (1 - alpha));

% Initial guess for consumption policy
policy_c = repmat((1 + r) * a_grid + w * mean(l_grid), 1, N);

% Utility functions
u = @(c) (c.^(1 - sigma)) / (1 - sigma);
u_prime = @(c) c.^(-sigma);
u_prime_inv = @(u_p) u_p.^(-1 / sigma);

dist = 1;
iter = 0;

while dist > tol && iter < max_iter
    iter = iter + 1;
    policy_c_next = zeros(size(policy_c)); % Updated consumption policy
    policy_k = zeros(size(policy_c));      % Asset policy

    % Step 1: Compute RHS of Euler equation
    RHS = zeros(grid_size, N);
    for j = 1:N
        for m = 1:N
            RHS(:, j) = RHS(:, j) + beta * (1 + r) * P(j, m) * u_prime(policy_c(:, m));
        end
    end

    % Step 2: Invert marginal utility to find consumption
    c_next = u_prime_inv(RHS);

    % Step 3: Compute endogenous grid
    for j = 1:N
        a_hat = (c_next(:, j) + a_grid - w * s(j)) / (1 + r);

        % Step 4: Interpolate the asset policy
        g_a_temp = interp1(a_hat, a_grid, a_grid, 'linear', 'extrap');

        % Step 5: Apply borrowing constraint
        g_a_temp(g_a_temp < amin) = amin; % Enforce borrowing constraint
        policy_k(:, j) = g_a_temp;             % Update the asset policy

        % Step 6: Recover the consumption policy
        policy_c_next(:, j) = (1 + r) * a_grid + w * s(j) - policy_k(:, j);
    end

    % Step 7: Check for convergence
    dist = max(max(abs(policy_c_next - policy_c)));
    policy_c = policy_c_next; % Update consumption policy
    
    fprintf('Iteration: %d, Distance: %.8e\n', iter, dist);
end

if iter == max_iter
    warning('Convergence not achieved!');
else
    disp('Convergence achieved!');
end

%% Simulation with Piecewise Linear Interpolation

num_simulations = 10000;  % Number of simulations
sim_z = zeros(num_simulations, 1);
sim_k = zeros(num_simulations, 1);
sim_c = zeros(num_simulations, 1); % Consumption
sim_y = zeros(num_simulations, 1); % Net income
sim_s = zeros(num_simulations, 1); % Savings
sim_gy = zeros(num_simulations, 1); % Gross Income
sim_z(1) = randi(N);
sim_k(1) = a_grid(randi(grid_size));

for t = 2:num_simulations
    % State transition
    sim_z(t) = find(rand < cumsum(P(sim_z(t - 1), :)), 1);

    % Interpolation for capital transition
    k_prev = sim_k(t - 1);
    k_policy = policy_k(:, sim_z(t));  % Policy function for current state

    % Use interp1 for piecewise linear interpolation
    sim_k(t) = interp1(a_grid, k_policy, k_prev, 'linear', 'extrap');

    % Interpolation for consumption
    c_policy = policy_c(:, sim_z(t));  % Policy function for consumption
    sim_c(t) = interp1(a_grid, c_policy, k_prev, 'linear', 'extrap');

    % Compute net income, gross income and savings
    sim_y(t) = r * sim_k(t) + w * s(sim_z(t));
    sim_gy(t) = r * sim_k(t) + w * s(sim_z(t)) + delta*sim_k(t);
    sim_s(t) = r * sim_k(t) + w * s(sim_z(t)) + delta*sim_k(t) - sim_c(t);
end

sim_k = sim_k(:);
sim_z = sim_z(:);

% Compute aggregate capital supply 
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

    dist = 1;
    iter = 0;
    
    % Recalculate EGM Iteration with updated r
    while dist > tol && iter < max_iter
        iter = iter + 1;
        policy_c_next = zeros(size(policy_c)); % Updated consumption policy
        policy_k = zeros(size(policy_c));      % Asset policy
        r = r_guess;

        % Step 1: Compute RHS of Euler equation
        RHS = zeros(grid_size, N);
        for j = 1:N
            for m = 1:N
                RHS(:, j) = RHS(:, j) + beta * (1 + r) * P(j, m) * u_prime(policy_c(:, m));
            end
        end
    
        % Step 2: Invert marginal utility to find consumption
        c_next = u_prime_inv(RHS);
    
        % Step 3: Compute endogenous grid
        for j = 1:N
            a_hat = (c_next(:, j) + a_grid - w * s(j)) / (1 + r);
    
            % Step 4: Interpolate the asset policy
            g_a_temp = interp1(a_hat, a_grid, a_grid, 'linear', 'extrap');
    
            % Step 5: Apply borrowing constraint
            g_a_temp(g_a_temp < amin) = amin; % Enforce borrowing constraint
            policy_k(:, j) = g_a_temp;             % Update the asset policy
    
            % Step 6: Recover the consumption policy
            policy_c_next(:, j) = (1 + r) * a_grid + w * s(j) - policy_k(:, j);
        end
    
        % Step 7: Check for convergence
        dist = max(max(abs(policy_c_next - policy_c)));
        policy_c = policy_c_next; % Update consumption policy
        
    end
    
    if iter == max_iter
        warning('Convergence not achieved!');
    else
        disp('Convergence achieved!');
    end
    
    % Simulate again with updated policy function
    for t = 2:num_simulations
        sim_z(t) = find(rand < cumsum(P(sim_z(t - 1), :)), 1);
        k_prev = sim_k(t - 1);
        k_policy = policy_k(:, sim_z(t));
        sim_k(t) = interp1(a_grid, k_policy, k_prev, 'linear', 'extrap');

        % Interpolation for consumption
        c_policy = policy_c(:, sim_z(t));  % Policy function for consumption
        sim_c(t) = interp1(a_grid, c_policy, k_prev, 'linear', 'extrap');

        % Compute net income, gross income and savings
        sim_y(t) = r * sim_k(t) + w * s(sim_z(t));
        sim_gy(t) = r * sim_k(t) + w * s(sim_z(t)) + delta*sim_k(t);
        sim_s(t) = r * sim_k(t) + w * s(sim_z(t)) + delta*sim_k(t) - sim_c(t);
    end

    sim_k = sim_k(:);
    sim_z = sim_z(:);
    
    agg_k_supply = mean(sim_k(:));
    k_supply(r_iter) = agg_k_supply;
    k_demand(r_iter) = labor*(alpha/(r_guess+delta))^(1/(1-alpha));
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

savings_percent = (delta*alpha / (r + delta))*100;
r_percent = r*100;

r_history = sort(r_history, "ascend");
k_demand = sort(k_demand ,"descend");
k_supply = sort(k_supply, "ascend");

%% Plotting

figure(1);
plot(k_demand,r_history, 'r', 'LineWidth', 2);
xlim([0, 18]);
hold on;
plot(k_supply,r_history, 'b--', 'LineWidth', 2);
xlim([0, 18]);
yline((1-beta)/beta);
xlim([0,18]);
xlabel('Total Assets');
ylabel('Interest Rate');
legend('Capital Demand', 'Capital Supply');
title('Steady State: mu=5, rho=0.6, sig=0.2');
grid on;

figure(2);
plot(a_grid,policy_k(:,1), 'r:', 'LineWidth', 2);
xlim([0, 20]);
ylim([0, 20]);
hold on;
plot(a_grid,policy_k(:,2), 'b--', 'LineWidth', 2);
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
[f_gy, xi_gy] = ksdensity(sim_gy);
[f_s, xi_s] = ksdensity(sim_s);

% Normalize the density estimates
f_w_normalized = f_w / (sum(f_w)); 
f_c_normalized = f_c / (sum(f_c)); 
f_y_normalized = f_y / (sum(f_y));
f_gy_normalized = f_gy / (sum(f_gy));
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
plot(xi_gy, f_gy_normalized, 'LineWidth', 2);
plot(xi_s, f_s_normalized, 'LineWidth', 2);
% hold off;    
title('Density Consumption, Net Income, Gross Income and Savings');
xlabel('Value');
ylabel('Density');
legend('Consumption', 'Net Income', 'Gross Income', 'Savings');
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
histogram(sim_gy, 'Normalization', 'probability');
title('Histogram of Gross Income');
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
sorted_gy = sort(sim_gy);
sorted_s = sort(sim_s);

% Compute the cumulative sum and the cumulative sum of population share
cumulative_k = cumsum(sorted_k) / sum(sorted_k);
cumulative_population_k = (1:length(sorted_k)) / length(sorted_k);

cumulative_c = cumsum(sorted_c) / sum(sorted_c);
cumulative_population_c = (1:length(sorted_c)) / length(sorted_c);

cumulative_y = cumsum(sorted_y) / sum(sorted_y);
cumulative_population_y = (1:length(sorted_y)) / length(sorted_y);

cumulative_gy = cumsum(sorted_gy) / sum(sorted_gy);
cumulative_population_gy = (1:length(sorted_gy)) / length(sorted_gy);

cumulative_s = cumsum(sorted_s) / sum(sorted_s);
cumulative_population_s = (1:length(sorted_s)) / length(sorted_s);

% Calculate the area under the Lorenz curve
lorenz_area_k = trapz(cumulative_population_k, cumulative_k);
lorenz_area_c = trapz(cumulative_population_c, cumulative_c);
lorenz_area_y = trapz(cumulative_population_y, cumulative_y);
lorenz_area_gy = trapz(cumulative_population_gy, cumulative_gy);
lorenz_area_s = trapz(cumulative_population_s, cumulative_s);

% Calculate the Gini coefficient
gini_coefficient_k = 1 - 2 * lorenz_area_k;
gini_coefficient_c = 1 - 2 * lorenz_area_c;
gini_coefficient_y = 1 - 2 * lorenz_area_y;
gini_coefficient_gy = 1 - 2 * lorenz_area_gy;
gini_coefficient_s = 1 - 2 * lorenz_area_s;

disp(['Gini Coefficient for Wealth: ', num2str(gini_coefficient_k)]);
disp(['Gini Coefficient for Consumption: ', num2str(gini_coefficient_c)]);
disp(['Gini Coefficient for Net Income: ', num2str(gini_coefficient_y)]);
disp(['Gini Coefficient for Gross Income: ', num2str(gini_coefficient_gy)]);
disp(['Gini Coefficient for Savings: ', num2str(gini_coefficient_s)]);

% Plot the Lorenz Curve
figure;
plot(cumulative_population_k, cumulative_k, "b-", 'LineWidth', 2);
hold on;
plot(cumulative_population_c, cumulative_c, "r-", 'LineWidth', 2);
plot(cumulative_population_y, cumulative_y, "y-", 'LineWidth', 2);
plot(cumulative_population_gy, cumulative_gy, "m-", 'LineWidth', 2);
plot(cumulative_population_s, cumulative_s, "g-", 'LineWidth', 2);
plot([0 1], [0 1], 'k--'); % Line of equality
xlabel('Cumulative Share of Population');
ylabel('Cumulative Share of Wealth');
title('Lorenz Curves');
legend('Wealth', 'Consumption', 'Net Income', 'Gross Income', 'Savings');
grid on;
