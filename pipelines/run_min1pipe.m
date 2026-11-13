function run_min1pipe(input_tif, outdir, fps, spatialr, se, ismc, min1pipe_dir, n_workers)
% run_min1pipe -- MIN1PIPE on one recording, headless, for the benchmark.
%
%   matlab -batch "addpath('C:/.../pipelines'); run_min1pipe('C:/.../rep1/synthetic_5min.tif', 'C:/.../rep1', 10, 1, 5, 1, 'C:/Users/ariAccount/Desktop/MIN1PIPE', 8)"
%
% Arguments
%   input_tif     multi-page TIFF made by bench/make_tiff.py, already copied into
%                 outdir by run_benchmark (MIN1PIPE writes its files next to it)
%   outdir        run folder; min1pipe_standard.mat is written here
%   fps           frame rate, passed as both Fsi and Fsi_new (no temporal downsampling)
%   spatialr      spatial downsampling ratio, 1 = none. Footprints are resized back
%                 to full resolution before saving if < 1.
%   se            neuron_size (half neuron size, px). Pass [] to let MIN1PIPE
%                 choose spatialr and se itself, as demo_min1pipe_HPC.m does.
%   ismc          1 = MIN1PIPE motion correction on
%   min1pipe_dir  MIN1PIPE repo folder
%   n_workers     parallel pool size, to match the other tools (8). 0 = no pool.
%
% Calls min1pipe_HPC(Fsi, Fsi_new, spatialr, se, ismc, flag, path_name, file_name),
% the dialog-free entry point (signature checked against JinghaoLu/MIN1PIPE master).
% MIN1PIPE is deliberately NOT added to the path here: its own min1pipe_init adds
% it and runs cvx_setup, but only when the folder is not already on the path.

if nargin < 8 || isempty(n_workers); n_workers = 8; end
stage_file = getenv('BENCH_STAGE_FILE');

[tdir, tbase, text] = fileparts(input_tif);
if ~strcmpi(text, '.tif') && ~strcmpi(text, '.tiff')
    error('run_min1pipe: give the TIFF made by bench/make_tiff.py, not %s', text);
end
if ~isfolder(fullfile(min1pipe_dir, 'utilities', 'cvx'))
    % MIN1PIPE would download CVX itself on first use, but that would land inside
    % a timed run. Install it once beforehand instead, e.g. from a command prompt:
    %   matlab -batch "unzip('http://web.cvxr.com/cvx/cvx-w64.zip', '<MIN1PIPE>/utilities'); run('<MIN1PIPE>/utilities/cvx/cvx_setup.m')"
    error(['run_min1pipe: CVX is not installed in %s/utilities/cvx. Install it once before ' ...
           'benchmarking (see the comment above this line in run_min1pipe.m).'], min1pipe_dir);
end

% --- parallel pool of fixed size (MIN1PIPE would otherwise open one with
%     feature('numCores') workers) ---
t0 = now_s();
if n_workers > 0 && license('test', 'Distrib_Computing_Toolbox')
    p = gcp('nocreate');
    if ~isempty(p) && p.NumWorkers ~= n_workers; delete(p); p = []; end
    if isempty(p); parpool(n_workers); end
end
write_stage(stage_file, 'parpool_start', t0, now_s());

% --- Curve Fitting Toolbox stand-in ---
% MIN1PIPE's motion correction calls smooth() (Curve Fitting Toolbox). If that
% toolbox is missing, use the moving-average stand-in in pipelines/compat.
used_smooth_shim = isempty(which('smooth'));
if used_smooth_shim
    addpath(fullfile(fileparts(mfilename('fullpath')), 'compat'));
    fprintf('[bench] Curve Fitting Toolbox not found: using compat/smooth.m (moving average)\n');
end

% --- run ---
t0 = now_s();
old = cd(min1pipe_dir);          % so min1pipe_HPC resolves and min1pipe_init sets up paths + CVX
cleanup = onCleanup(@() cd(old));
flag = 1;                         % automatic seed selection
fname = min1pipe_HPC(fps, fps, spatialr, se, ismc, flag, [tdir filesep], [tbase text]);
write_stage(stage_file, 'min1pipe', t0, now_s());

% --- export to the benchmark's standard form ---
t0 = now_s();
R = load(fname, 'roifn', 'sigfn', 'spkfn', 'pixh', 'pixw');
info = imfinfo(input_tif);
H = info(1).Height; W = info(1).Width;
n = size(R.roifn, 2);
roi = reshape(full(R.roifn), R.pixh, R.pixw, n);
if R.pixh ~= H || R.pixw ~= W
    roi = imresize(roi, [H W], 'bilinear');
    roi(roi < 0) = 0;
end
A = single(permute(roi, [3 1 2]));   % (n, H, W)
C = single(R.sigfn);
S = single(R.spkfn);
if size(C, 1) ~= n; C = C'; end
if size(S, 1) ~= n; S = S'; end
used_spatialr = R.pixh / H;
save(fullfile(outdir, 'min1pipe_standard.mat'), 'A', 'C', 'S', 'H', 'W', 'used_spatialr', 'used_smooth_shim', '-v7');

% remove the large intermediates and the input copy
delete(input_tif);
for pat = {'*_frame_all.mat', '*_reg.mat'}
    d = dir(fullfile(tdir, pat{1}));
    for k = 1:numel(d); delete(fullfile(tdir, d(k).name)); end
end
write_stage(stage_file, 'export', t0, now_s());
fprintf('[bench] saved %d units, %d frames (MIN1PIPE pixh=%d)\n', n, size(C, 2), R.pixh);
end

function t = now_s()
t = posixtime(datetime('now'));
end

function write_stage(path, name, t0, t1)
fprintf('[bench] stage %-24s %.1fs\n', name, t1 - t0);
if isempty(path); return; end
fid = fopen(path, 'a');
fprintf(fid, '{"stage": "%s", "start": %.3f, "end": %.3f, "dur_s": %.3f}\n', name, t0, t1, t1 - t0);
fclose(fid);
end
