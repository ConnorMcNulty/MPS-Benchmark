function ys = smooth(y, span, method)
% smooth -- stand-in for the Curve Fitting Toolbox smooth(), moving-average
% method only, used ONLY when that toolbox is not installed (run_min1pipe.m adds
% this folder to the path only if which('smooth') finds nothing).
%
% Matches the toolbox's documented 'moving' behaviour:
%   * default span 5; an even span is reduced by 1
%   * near the ends the window shrinks symmetrically, so ys(1) = y(1) and
%     ys(end) = y(end)
%   * returns a column vector
% MIN1PIPE calls it once, in utilities/movement_correction/movement_thres.m,
% as smooth(y, span).
if nargin < 2 || isempty(span); span = 5; end
if nargin >= 3 && ~strcmpi(method, 'moving')
    error('compat smooth: only the ''moving'' method is implemented');
end
y = y(:);
n = numel(y);
if span < 1; span = ceil(span * n); end   % fraction of the data length
span = max(1, floor(span));
if mod(span, 2) == 0; span = span - 1; end
h = (span - 1) / 2;
ys = zeros(n, 1);
cs = [0; cumsum(y)];
for i = 1:n
    k = min([h, i - 1, n - i]);
    ys(i) = (cs(i + k + 1) - cs(i - k)) / (2 * k + 1);
end
end
