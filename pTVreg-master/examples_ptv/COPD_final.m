addpath(genpath('../mutils/My/'));
addpath(genpath('../ptv'));
bpath = 'D:\OneDrive\OneDrive - The Hong Kong Polytechnic University\PolyU\DIR\COPD\';

TRE_e = zeros(2, 2, 10);
TIME_e = zeros(2, 2, 10);
fast_lcc = 1;
use_refinement = 1;
for copd_idx = 1:1
    [volmov, volfix, pts_mov, pts_fix, spc, volsz] = get_COPD_data(bpath, copd_idx);
    volmov = img_thr(volmov, 80, 1200, 1);
    volfix = img_thr(volfix, 80, 1200, 1);
    spc = [1 1 1];
    % crop images 
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

    bszv = size(volmov);
    spc_orig = spc;
%     spc_tmp = [1, 1, 1];
    volfix = volresize(volfix, round(bszv .* spc .* spc_tmp), 1);
    volmov = volresize(volmov, round(bszv .* spc .* spc_tmp), 1);
    spc = [1,1,1] ./ spc_tmp;
    
    % configure registration
    opts = [];
    opts.grid_spacing = [4, 4, 4];  
    opts.cp_refinements = 0;
    opts.loc_cc_approximate = fast_lcc;
    
    opts.display = 'off';
    opts.k_down = 0.7;
    opts.interp_type = 0;
    opts.nlvl = []; 
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

    timer = tic();
    [voldef, Tptv, Kptv] = ptv_register(volmov, volfix, opts);
    TIME_e(use_refinement+1, fast_lcc+1, copd_idx)=toc(timer);

    voldef_rsz = volresize(voldef, bszv);
    Tptv_rsz = cat(4, volresize(Tptv(:,:,:,1), bszv), volresize(Tptv(:,:,:,2), bszv), volresize(Tptv(:,:,:,3), bszv)); 
    [~, Tptv_rsz] = uncrop_data(voldef_rsz, Tptv_rsz, crop_v, init_size);
    [pt_errs_phys, pts_moved_pix, TRE_phys, TREstd_phys] = DIR_movepoints(pts_mov, pts_fix, Tptv_rsz, spc_orig, []);
    TRE_e(use_refinement+1, fast_lcc+1, copd_idx) = mean(TRE_phys);
    fprintf('TRE: %f.\n', mean(TRE_phys))
    % TRE before
    Tptv_zeros = zeros(size(Tptv));
    [~, Tptv_rsz] = uncrop_data(voldef, Tptv_zeros, crop_v, init_size);
    [pt_errs_phys, pts_moved_pix, TRE_phys, TREstd_phys] = DIR_movepoints(pts_mov, pts_fix, Tptv_rsz, spc_orig, []);
    TREs(idx) = mean(TRE_phys);
    fprintf('TRE_before: %f.\n', mean(TRE_phys))
    TRE_e(use_refinement+1, resize+1, fast_lcc+1, idx) = mean(TRE_phys);
    folder_name = ['case_',char(string(idx))];
    mkdir(['COPD/',folder_name])
    mov_name = 'volmov';
    fix_name = 'volfix';
    dvf_name = 'dvf';
    save(['COPD/',folder_name,'/',mov_name,'.mat'], 'volmov');
    save(['COPD/',folder_name,'/',fix_name,'.mat'], 'volfix');
    save(['COPD/',folder_name,'/',dvf_name,'.mat'], 'Tptv');
end
