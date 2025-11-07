clc;
clear all;
close all;

%% Reading data files


filename = 'all_slosh_data.mat';

slosh_results = load(filename).all_data;

%% Calculating intermediates

% Calculate gamma_k
N = size(slosh_results.c_k, 1);
num_modes_to_calculate = N;

gamma_k = zeros(num_modes_to_calculate, 1);
a = slosh_results.radius_at_surface;
V = slosh_results.volume;
B = slosh_results.B_matrix;
C = slosh_results.c_k;
g = 9.81;
l_O3 = 0;


for k = 1:num_modes_to_calculate
    ck = C(:, k); % Eigenvector for mode k
    double_sum = ck' * B * ck;
    gamma_k(k) = (pi * a^3 / V) * double_sum;
end

disp('Calculated gamma_k:');
disp(gamma_k);

% Calculute b*_k
bk_star = zeros(num_modes_to_calculate, 1);
first_row_B = B(1, :); % Get the first row of B

for k = 1:num_modes_to_calculate
    ck = C(:, k); % Eigenvector for mode k
    summation = dot(ck, first_row_B); % Sum of c_n^k * b_1n
    if gamma_k(k) ~= 0 % Avoid division by zero
        bk_star(k) = (pi * a^3 / (V * gamma_k(k))) * summation;
    else
        bk_star(k) = NaN; % Indicate an issue if gamma_k is zero
    end
end

disp('Calculated bk_star:');
disp(bk_star);

% Calculate h_k
hk = zeros(num_modes_to_calculate, 1);
lambda_k = diag(slosh_results.lambda_k);
s_n_vector = slosh_results.s_n_vector;

for k = 1:num_modes_to_calculate
    ck = C(:, k); % Eigenvector for mode k
    summation = dot(ck, s_n_vector); % Sum of c_n^k * s_n
    denominator = V * gamma_k(k) * lambda_k(k);
    if denominator ~= 0 % Avoid division by zero
        hk(k) = (2 * pi * a^3 / denominator) * summation;
    else
        hk(k) = NaN; % Indicate an issue
    end
end

disp('Calculated hk:');
disp(hk);

%% Defining the basis functions

% --- Phase 3: Reconstruct the Flow Field ---

fprintf('Phase 3, Step 1: Defining Basis Functions...\n');
    
C = slosh_results.c_k; % Eigenvector matrix (sorted)
N = size(C, 1);       % Number of basis functions (should be 10)
M = 5;                % Number of shallow tank functions (as per paper)
L_nondim = slosh_results.dist_CoM_surface_nondim; % Non-dim L = l/a

% --- Define Roots for Deep Tank Functions ---
% These are the first N-M (e.g., 10-5 = 5) roots of J1'(x) = 0
% Refer: https://mathworld.wolfram.com/BesselFunctionZeros.html
j_roots_vec = [1.84118, 5.33144, 8.53632, 11.7060, 14.8636]; 

% Pad with zeros to make indexing easy (j_roots(n) will be 0 for n<=M)
j_roots = [zeros(1, M), j_roots_vec(1:min(N-M, length(j_roots_vec)))];

% --- Create the Basis Function Handle (phi_hat_n) ---
% This function implements Equation 9
% It takes non-dimensional R, Z, and the mode index 'n'
phi_hat_n = @(R, Z, n) ...
    (n <= M) * (R.^(2*n - 1)) + ...  % Shallow tank part
    (n > M)  * (besselj(1, j_roots(n) * R) .* exp(j_roots(n) * (Z - L_nondim))); % Deep tank part

fprintf('Basis function handle "phi_hat_n(R, Z, n)" is now defined.\n');

%% Creating a mesh

fprintf('Phase 3, Step 2: Creating 2D Non-Dimensional Mesh Grid...\n');

% --- Define Grid Resolution ---
num_points_R = 32; % Resolution in R (radial)
num_points_Z = 32; % Resolution in Z (axial)
num_points_theta = 32;

% --- Get Non-Dimensional Geometric Boundaries from Saved Data ---
% 'a' is the radius at the free surface
a = slosh_results.radius_at_surface; 

% 'slosh_results.tank_boundaries_CM(3)' is the dimensional z_min
% We divide by 'a' to make it non-dimensional
Z_min_nd = slosh_results.tank_boundaries_CM(3) / a; 

% 'L_nondim' is the non-dimensional height of the free surface (Z_max)
Z_max_nd = slosh_results.dist_CoM_surface_nondim; % This is 'L' from Eq. 9

% 'slosh_results.epsilon' is the non-dimensional inner radius at the surface (R_min)
R_min_nd = slosh_results.epsilon;
R_max_nd = 1.0; % Outer radius at the surface is always 1.0 by definition

% --- Create Vectors and Meshgrid ---
R_vec_nd = linspace(R_min_nd, R_max_nd, num_points_R);
Z_vec_nd = linspace(Z_min_nd, Z_max_nd, num_points_Z);

[R_grid_nd, Z_grid_nd] = meshgrid(R_vec_nd, Z_vec_nd);

fprintf('Non-dimensional grid (R_grid_nd, Z_grid_nd) created.\n');

%% --- BATCH DATA GENERATION ---

% -------------------------------------------------------------------------
% %% STATIC SETUP (Run Once)
% We do all calculations that are independent of the forces *before* the loop.
% -------------------------------------------------------------------------
fprintf('--- STARTING STATIC SETUP ---\n');

% --- 1. Get Valid Mode Data ---
lambda_k_all_vec = diag(slosh_results.lambda_k);
omega_k_sorted = slosh_results.natural_frequencies_rads;
valid_indices = find(lambda_k_all_vec > 1e-6 & imag(omega_k_sorted) == 0);
num_valid_modes = length(valid_indices);
C_valid = slosh_results.c_k(:, valid_indices); % Eigenvectors for valid modes
fprintf('Identified %d valid physical modes.\n', num_valid_modes);

% --- 2. Reconstruct Static Spatial Shapes (Dimensional) ---
% These 2D dimensional shapes Phi_k(r,z) are constant for all runs
phi_k_spatial = zeros(num_points_Z, num_points_R, num_valid_modes);
for i = 1:num_valid_modes
    current_mode_index = valid_indices(i);
    phi_k_mode_i = zeros(num_points_Z, num_points_R); 
    ck_i = C_valid(:, i); 
    for n = 1:N
        basis_shape_n = phi_hat_n(R_grid_nd, Z_grid_nd, n);
        phi_k_mode_i = phi_k_mode_i + ck_i(n) * basis_shape_n;
    end
    % Scale to dimensional potential shape Phi_k
    phi_k_spatial(:,:,i) = (a * g) / (omega_k_sorted(current_mode_index)) * phi_k_mode_i;
end
fprintf('Static dimensional spatial modes (phi_k_spatial) reconstructed.\n');

% --- 3. Get Static Scaling Factors and Grids ---
l_dim = slosh_results.dist_CoM_surface_dim;
lambda_k_valid = lambda_k_all_vec(valid_indices);
omega_k_valid = omega_k_sorted(valid_indices);

% This is the key scaling factor: dot_delta_k -> lambda_k(t)
scaling_factors = (l_dim .* omega_k_valid) ./ (a * g .* lambda_k_valid);

% Define 3D grid parameters [H, W, D]
num_H = num_points_Z;     % Height (z)
num_W = num_points_R;     % Width (r)
num_D = num_points_theta; % Depth (theta)
num_C_vel = 3;            % 3 velocity channels
num_C_force = 3;          % 3 force channels
num_C_total = num_C_vel + num_C_force; % 6 total channels

% Create static grids needed for the loop
theta_vec = linspace(0, 2*pi, num_D);
r_vec_dim = a * R_vec_nd;
z_vec_dim = a * Z_vec_nd; 
[r_grid, ~, theta_grid] = meshgrid(r_vec_dim, z_vec_dim, theta_vec);
sin_theta_grid = sin(theta_grid);

fprintf('Static setup complete. Ready for batch loop.\n');

% -------------------------------------------------------------------------
% %% BATCH SIMULATION LOOP
% This loop will run 'num_simulations' times, creating one file per run.
% -------------------------------------------------------------------------

num_simulations = 1000; % <<<--- SET HOW MANY FILES YOU WANT TO GENERATE
output_directory = './FNO_Dataset_6_Channel/';
if ~exist(output_directory, 'dir')
   mkdir(output_directory)
end

for run_index = 1:num_simulations
    fprintf('\n--- Starting Simulation Run %d / %d ---\n', run_index, num_simulations);
    
    % --- 1. Define Randomized Inputs for THIS run ---
    t_span = [0 2]; 
    
    % Randomize a pulse for alpha2_func
    mag_alpha2 = 2.0 + rand() * 8.0;  % Random magnitude between 2.0 and 10.0
    dur_alpha2 = 0.5 + rand() * 1.0;  % Random duration between 0.5 and 2.0s
    start_alpha2 = rand() * (t_span(end) - dur_alpha2); % Random start time
    
    alpha2_func = @(t) (t >= start_alpha2 && t <= (start_alpha2 + dur_alpha2)) * mag_alpha2;
    
    % --- (Optional: Add randomness to other forces) ---
    const_alpha3 = 9.81; % We'll keep this constant for now, but you could randomize it too
    alpha3_func = @(t) const_alpha3;
    ddot_theta_func = @(t) 0; % Keep this zero for now
    
    fprintf('  Run %d: alpha2_func = %.1f mag, %.1fs dur, at %.1fs\n', ...
            run_index, mag_alpha2, dur_alpha2, start_alpha2);

    % --- 2. Solve ODEs for THIS run's forces ---
    num_all_modes = N;
    delta_results_all = cell(num_all_modes, 1);
    dot_delta_results_all = cell(num_all_modes, 1);
    time_results_all = cell(num_all_modes, 1);
    initial_conditions = [0; 0]; 

    for i = 1:num_all_modes
        lambda_k_i = lambda_k_all_vec(i);
        bk_star_i = bk_star(i);
        hk_i = hk(i);

        forcing_term = @(t) -lambda_k_i * bk_star_i * alpha2_func(t) ...
                           -lambda_k_i * (l_O3 * bk_star_i - l_dim * (bk_star_i - hk_i)) * ddot_theta_func(t);
        
        ode_func = @(t, y) [y(2); 
                           forcing_term(t) - alpha3_func(t) * (lambda_k_i / L_nondim) * y(1)]; 
        try
            if lambda_k_i <= 1e-9 
                 t_out = linspace(t_span(1), t_span(end), 50).'; y_out = NaN(50, 2);
            else
                [t_out, y_out] = ode45(ode_func, t_span, initial_conditions);
            end
            time_results_all{i} = t_out;
            delta_results_all{i} = y_out(:, 1);  
            dot_delta_results_all{i} = y_out(:, 2);
        catch ME_ode
            fprintf('  ODE solver failed for mode %d: %s\n', i, ME_ode.message);
            time_results_all{i} = linspace(t_span(1), t_span(end), 50).';
            delta_results_all{i} = NaN(50, 1); dot_delta_results_all{i} = NaN(50, 1);
        end
    end
    fprintf('  ODEs solved for run %d.\n', run_index);
    
    % --- 3. Time-Stepping Reconstruction for THIS run ---
    dt_save = 0.05; % Save one snapshot every 0.05 seconds
    save_time_vector = (t_span(1):dt_save:t_span(end)).';
    num_time_steps = length(save_time_vector);

    % Pre-allocate the 6-CHANNEL 5D array
    model_data_5D = zeros(num_time_steps, num_C_total, num_H, num_W, num_D);

    % Evaluate force functions for all time steps ONCE
    alpha2_data_vec = arrayfun(alpha2_func, save_time_vector);
    alpha3_data_vec = arrayfun(alpha3_func, save_time_vector);
    ddot_theta_data_vec = arrayfun(ddot_theta_func, save_time_vector);
    
    % Create static 3D force grids ONCE
    % (ones(num_H, num_W, num_D) is slow, repmat is faster)
    grid_shape = [num_H, num_W, num_D];
    alpha2_grid = repmat(alpha2_data_vec(1), grid_shape); % Placeholder
    alpha3_grid = repmat(alpha3_data_vec(1), grid_shape); % Placeholder
    ddot_theta_grid = repmat(ddot_theta_data_vec(1), grid_shape); % Placeholder

    for t_idx = 1:num_time_steps
        t_current = save_time_vector(t_idx);
        
        % --- 3a. Interpolate dot_delta_k at t_current ---
        dot_delta_k_at_t = zeros(num_valid_modes, 1);
        for i = 1:num_valid_modes
            mode_index = valid_indices(i);
            if ~all(isnan(dot_delta_results_all{mode_index}))
                dot_delta_k_at_t(i) = interp1(time_results_all{mode_index}, ...
                                              dot_delta_results_all{mode_index}, ...
                                              t_current, 'spline', 'extrap');
            else
                dot_delta_k_at_t(i) = 0;
            end
        end
        
        lambda_k_at_t = dot_delta_k_at_t .* scaling_factors;
        
        % --- 3b. Reconstruct Total Potential (Phi) at t_current ---
        Phi_total = zeros(num_H, num_W, num_D);
        for i = 1:num_valid_modes
            Phi_k_3D = repmat(phi_k_spatial(:,:,i), [1, 1, num_D]);
            lambda_t_i = lambda_k_at_t(i);
            Phi_total = Phi_total + lambda_t_i * Phi_k_3D .* sin_theta_grid;
        end
        
        % --- 3c. Calculate Velocity Field (V) at t_current ---
        [dPhi_dr, dPhi_dz, dPhi_dtheta] = gradient(Phi_total, z_vec_dim, r_vec_dim, theta_vec);
        
        Vr = dPhi_dr;
        Vz = dPhi_dz;
        Vtheta = (1 ./ r_grid) .* dPhi_dtheta;
        Vtheta(r_grid < 1e-6) = 0;
        
        % --- 3d. Update 3D Force Grids for t_current ---
        alpha2_grid(:) = alpha2_data_vec(t_idx);
        alpha3_grid(:) = alpha3_data_vec(t_idx);
        ddot_theta_grid(:) = ddot_theta_data_vec(t_idx);

        % --- 3e. Store ALL 6 Channels in 5D Array ---
        model_data_5D(t_idx, 1, :, :, :) = Vr;
        model_data_5D(t_idx, 2, :, :, :) = Vtheta;
        model_data_5D(t_idx, 3, :, :, :) = Vz;
        model_data_5D(t_idx, 4, :, :, :) = alpha2_grid;
        model_data_5D(t_idx, 5, :, :, :) = alpha3_grid;
        model_data_5D(t_idx, 6, :, :, :) = ddot_theta_grid;
    end
    fprintf('  6-Channel 5D data reconstructed for run %d.\n', run_index);

    % --- 4. Save THIS run's data ---
    fprintf('  Saving data for run %d...\n', run_index);
    
    [X_grid, Y_grid, z_grid] = meshgrid(r_vec_dim, z_vec_dim, theta_vec);
    X_grid = X_grid .* cos(theta_grid);
    Y_grid = Y_grid .* sin(theta_grid);
    
    filename_out = sprintf('%s/FNO_dataset_run_%04d.mat', output_directory, run_index);

    try
        save(filename_out, ...
             'model_data_5D', ...      % The 6-Channel 5D data [T, 6, H, W, D]
             'X_grid', 'Y_grid', 'z_grid', ... % The 3D coordinates
             'save_time_vector', ...       % The time vector (T)
             'alpha2_data_vec', ...        % 1D Input force 1
             'alpha3_data_vec', ...        % 1D Input force 2
             'ddot_theta_data_vec');
         
        fprintf('  Successfully saved: %s\n', filename_out);
    catch ME
        warning('  Failed to save run %d: %s\n', run_index, ME.message);
    end
end % --- End of batch loop ---

fprintf('--- BATCH DATA GENERATION COMPLETE ---\n');