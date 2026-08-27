addpath(genpath('mutils/My/'));
addpath(genpath('ptv/'));
% basepth = '../../../data_prj/dir_dataset/DIR_files/';
basepth ='D:\OneDrive\OneDrive - The Hong Kong Polytechnic University\PolyU\DIR\';


TIME_e = zeros(2,2,2,10); 
TRE_e = zeros(2,2,2,10);

% Demonstrate the effect of refinement. Not much difference
use_refinement = 0;
% Demonstrate the effect of resizing. Some contribution
resize = 0;
% Fast LCC. Much contribution
fast_lcc = 1;

for idx = 6:6
    % load the landmarks, extreme and 4D
    pts_struct = DIR_get_all_points_for_the_case(idx, basepth);
    % load the CT volumes. The spacing and size needs to be entered in the
    % function. Here only phase 00% and 50% were loaded
    % Can be adjusted by changing numbers
    [volmov, spc] = read_DIR_volume_4dCT(idx, 0, basepth);
    volmov = double(volmov);
    pts_mov = pts_struct.extreme.b;
%     pts_mov = pts_struct.smp{1,6}.pts;
    [volfix, spc] = read_DIR_volume_4dCT(idx, 5, basepth);
    volfix = double(volfix);
    pts_fix = pts_struct.extreme.e;
%     pts_fix = pts_struct.smp{1,1}.pts;
    
    % Image normalization to 0~1. Window between 80-900 (-920 to -100)
    volmov = img_thr(volmov, 80, 1200, 1);
    volfix = img_thr(volfix, 80, 1200, 1);
    
    % crop images 
    % crop the image according to the landmark coverage
    % 10 margin in 1st and 2nd dim, 5 margin in 3rd dim
    init_size = size(volmov);
    min_max1 = [ min(pts_mov, [], 1)', max(pts_mov, [], 1)'];
    min_max2 = [ min(pts_fix, [], 1)', max(pts_fix, [], 1)'];
    min_max = [ min(min_max1(:, 1), min_max2(:, 1)), max(min_max1(:, 2), min_max2(:, 2))];
    d = [10, 10, 5];
    crop_v = [ max(1, min_max(1,1) - d(1)), min(size(volmov, 1), min_max(1,2) + d(1)); ... 
               max(1, min_max(2,1) - d(2)), min(size(volmov, 2), min_max(2,2) + d(2)); ... 
               max(1, min_max(3,1) - d(3)), min(size(volmov, 3), min_max(3,2) + d(3));];
    volmov = crop_data(volmov, crop_v);
    volfix = crop_data(volfix, crop_v);
    spc = [1,1,1];
    % blurring data
%     volmov = imresize3(volmov, round(size(volfix)./2));
%     volmov = imresize3(volmov, round(size(volfix)));
    % isotropical resampling
    spc_orig = spc;
    % configure registration
    opts = [];
    opts.loc_cc_approximate = fast_lcc;
    opts.grid_spacing = [4, 4, 3];  % grid spacing in pixels
    opts.cp_refinements = 0;
    opts.display = 'off';
    opts.k_down = 0.7;
    opts.interp_type = 0;
    opts.metric = 'loc_cc_fftn_gpu';
    opts.metric_param = [1,1,1] * 2.1;
    opts.scale_metric_param = true;
    opts.isoTV = 0.11;
    opts.csqrt = 5e-3;
    opts.spline_order = 1;
    opts.border_mask = 5;
    opts.max_iters =  80;
    opts.check_gradients = 100*0;
    opts.pix_resolution = spc;

    timer = tic;
    % Tptv is the DVF
    [voldef, Tptv, Kptv] = ptv_register(volmov, volfix, opts);
    TIME_e(use_refinement+1, resize+1, fast_lcc+1, idx) = toc(timer);
    Tptv_rsz = Tptv;
    [~, Tptv_rsz] = uncrop_data(voldef, Tptv_rsz, crop_v, init_size);
    % move points and measure TRE
    [pt_errs_phys, pts_moved_pix, TRE_phys, TREstd_phys] = DIR_movepoints(pts_mov, pts_fix, Tptv_rsz, spc_orig, []);
    TREs(idx) = mean(TRE_phys);
    fprintf('TRE: %f.\n', mean(TRE_phys))
    % TRE before
    Tptv_zeros = zeros(size(Tptv));
    [~, Tptv_rsz] = uncrop_data(voldef, Tptv_zeros, crop_v, init_size);
    [pt_errs_phys, pts_moved_pix, TRE_phys, TREstd_phys] = DIR_movepoints(pts_mov, pts_fix, Tptv_rsz, spc_orig, []);
    TREs(idx) = mean(TRE_phys);
    fprintf('TRE_before: %f.\n', mean(TRE_phys))
    TRE_e(use_refinement+1, resize+1, fast_lcc+1, idx) = mean(TRE_phys);
    folder_name = ['case_',char(string(idx))];
    mkdir(['Group2_reverse/',folder_name])
    mov_name = 'volmov';
    fix_name = 'volfix';
    dvf_name = 'dvf';
    save(['Group2_reverse/',folder_name,'/',mov_name,'.mat'], 'volmov');
    save(['Group2_reverse/',folder_name,'/',fix_name,'.mat'], 'volfix');
    save(['Group2_reverse/',folder_name,'/',dvf_name,'.mat'], 'Tptv');
end

% for idx = 1:10
%     % load the landmarks, extreme and 4D
%     pts_struct = DIR_get_all_points_for_the_case(idx, basepth);
%     % load the CT volumes. The spacing and size needs to be entered in the
%     % function. Here only phase 00% and 50% were loaded
%     % Can be adjusted by changing numbers
%     [volmov, spc] = read_DIR_volume_4dCT(idx, 5, basepth);
%     volmov = double(volmov);
%     pts_mov = pts_struct.extreme.e;
% %     pts_mov = pts_struct.smp{1,6}.pts;
%     [volfix, spc] = read_DIR_volume_4dCT(idx, 0, basepth);
%     volfix = double(volfix);
%     pts_fix = pts_struct.extreme.b;
% %     pts_fix = pts_struct.smp{1,1}.pts;
%     
%     % Image normalization to 0~1. Window between 80-900 (-920 to -100)
%     volmov = img_thr(volmov, 80, 1200, 1);
%     volfix = img_thr(volfix, 80, 1200, 1);
%     
%     % crop images 
%     % crop the image according to the landmark coverage
%     % 10 margin in 1st and 2nd dim, 5 margin in 3rd dim
%     init_size = size(volmov);
%     min_max1 = [ min(pts_mov, [], 1)', max(pts_mov, [], 1)'];
%     min_max2 = [ min(pts_fix, [], 1)', max(pts_fix, [], 1)'];
%     min_max = [ min(min_max1(:, 1), min_max2(:, 1)), max(min_max1(:, 2), min_max2(:, 2))];
%     d = [10, 10, 5];
%     crop_v = [ max(1, min_max(1,1) - d(1)), min(size(volmov, 1), min_max(1,2) + d(1)); ... 
%                max(1, min_max(2,1) - d(2)), min(size(volmov, 2), min_max(2,2) + d(2)); ... 
%                max(1, min_max(3,1) - d(3)), min(size(volmov, 3), min_max(3,2) + d(3));];
%     volmov = crop_data(volmov, crop_v);
%     volfix = crop_data(volfix, crop_v);
%     spc = [1,1,1];
%     % blurring data
% %     volmov = imresize3(volmov, round(size(volfix)./2));
% %     volmov = imresize3(volmov, round(size(volfix)));
%     % isotropical resampling
%     spc_orig = spc;
%     % configure registration
%     opts = [];
%     opts.loc_cc_approximate = fast_lcc;
%     opts.grid_spacing = [4, 4, 3];  % grid spacing in pixels
%     opts.cp_refinements = 0;
%     opts.display = 'off';
%     opts.k_down = 0.7;
%     opts.interp_type = 0;
%     opts.metric = 'loc_cc_fftn_gpu';
%     opts.metric_param = [1,1,1] * 2.1;
%     opts.scale_metric_param = true;
%     opts.isoTV = 0.11;
%     opts.csqrt = 5e-3;
%     opts.spline_order = 1;
%     opts.border_mask = 5;
%     opts.max_iters =  80;
%     opts.check_gradients = 100*0;
%     opts.pix_resolution = spc;
% 
%     timer = tic;
%     % Tptv is the DVF
%     [voldef, Tptv, Kptv] = ptv_register(volmov, volfix, opts);
%     TIME_e(use_refinement+1, resize+1, fast_lcc+1, idx) = toc(timer);
%     Tptv_rsz = Tptv;
%     [~, Tptv_rsz] = uncrop_data(voldef, Tptv_rsz, crop_v, init_size);
%     % move points and measure TRE
%     [pt_errs_phys, pts_moved_pix, TRE_phys, TREstd_phys] = DIR_movepoints(pts_mov, pts_fix, Tptv_rsz, spc_orig, []);
%     TREs(idx) = mean(TRE_phys);
%     fprintf('TRE: %f.\n', mean(TRE_phys))
%     % TRE before
%     Tptv_zeros = zeros(size(Tptv));
%     [~, Tptv_rsz] = uncrop_data(voldef, Tptv_zeros, crop_v, init_size);
%     [pt_errs_phys, pts_moved_pix, TRE_phys, TREstd_phys] = DIR_movepoints(pts_mov, pts_fix, Tptv_rsz, spc_orig, []);
%     TREs(idx) = mean(TRE_phys);
%     fprintf('TRE_before: %f.\n', mean(TRE_phys))
%     TRE_e(use_refinement+1, resize+1, fast_lcc+1, idx) = mean(TRE_phys);
%     folder_name = ['case_',char(string(idx)),'_r'];
%     mkdir(['Group2_reverse/',folder_name])
%     mov_name = 'volmov';
%     fix_name = 'volfix';
%     dvf_name = 'dvf';
%     save(['Group2_reverse/',folder_name,'/',mov_name,'.mat'], 'volmov');
%     save(['Group2_reverse/',folder_name,'/',fix_name,'.mat'], 'volfix');
%     save(['Group2_reverse/',folder_name,'/',dvf_name,'.mat'], 'Tptv');
% end
errors = [];
num_points = [];
Tptv_rsz = -pred_dvf;
[~, Tptv_rsz] = uncrop_data(voldef, Tptv_rsz, crop_v, init_size);
for i = 1:10
    pts_mov_temp = pts_mov;
    pts_fix_temp = pts_fix;
    upper_limit = 101/10 * i;
    points = find(pts_mov(:,3) > upper_limit);
    num_points = [num_points; size(points)];
    pts_mov_temp(points,:) = [];
    pts_fix_temp(points,:) = [];
    [pt_errs_phys, pts_moved_pix, TRE_phys, TREstd_phys] = DIR_movepoints(pts_mov_temp, pts_fix_temp, Tptv_rsz, spc_orig, []);
    errors = [errors; TRE_phys];
end
