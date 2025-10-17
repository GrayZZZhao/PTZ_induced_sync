function [starts, ends_, durations] = compute_bursts(logic_array, fs, min_dur_s) % logic_array: 1xN logical (true during burst) % fs: samples/sec % min_dur_s: minimum burst length in seconds (optional)
x = logical(logic_array(:)).';              % row vector
d = diff([false, x, false]);                % rising/falling edges
starts    = find(d ==  1);
ends_     = find(d == -1) - 1;
durations = ends_ - starts + 1;             % samples

if nargin >= 3 && ~isempty(min_dur_s)
    keep = durations >= round(min_dur_s * fs);
    starts    = starts(keep);
    ends_     = ends_(keep);
    durations = durations(keep);
end
