clear; clc;

room_name = "ListeningRoom";
src_idx = 1;
height = 1.0;

idxX_pick = 16;
idxY_pick = 16;

filename = sprintf('ISOBEL_RTFs/%s_RTFs/source_%d/h_%d/idxX_%d_idxY_%d.mat', ...
    room_name, src_idx, height * 100, idxX_pick, idxY_pick);
d = load(filename);

RTF = d.RTF(:);
freqs = d.freqs;
rx_pos = d.rx_pos;

K = numel(RTF);
Fs = 1000; % sampling freq
duration = 1.0;
N = Fs * duration;
df = 1 / duration;

assert(N == 2*K, 'N must equal 2*K for this layout');

% build the full spectrum
H_full = zeros(N ,1);
H_full(1) = 0; % set DC to 0
H_full(2:K) = RTF(2:K); % positive frequencies
H_full(K+1) = 0; % nyquist bin
H_full(K+2:N) = conj(flipud(RTF(2:K))); % mirror conjugate for neg freqs

% ifft to time domain
h = ifft(H_full, 'symmetric');

fprintf('Receiver pos: [%.2f, %.2f, %.2f]\n', rx_pos);
fprintf('IR length: %d samples (%.3f s at Fs=%d Hz)\n', length(h), length(h) / Fs, Fs);
fprintf('Max |imag|: %.2e\n', max(abs(imag(ifft(H_full)))));
fprintf('Max |h|: %.4e\n', max(abs(h)));
fprintf('Mean(h): %.4e\n', mean(h));
fprintf('Any NaN/Inf: %d\n', any(~isfinite(h)));

% plots
t = (0:N-1) / Fs;

figure('Position', [100 100 1000 700]);

% time-domain IR
subplot(3,1,1);
plot(t, h, 'b'); hold on;
% expected exponential decay envelope from T60
T60 = 0.6;
delta = 3 * log(10) / T60;
env = max(abs(h)) * exp(-delta * t);
plot(t, env, 'r--', 'LineWidth', 1);
plot(t, -env, 'r--', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Amplitude');
title(sprintf('Impulse response — %s, src %d, rcv [%.1f, %.1f, %.1f]', ...
    room_name, src_idx, rx_pos));
legend('h(t)', sprintf('exp(-t·3ln10/T60), T60=%.2fs', T60));
grid on;

% zoomed in onset
subplot(3,1,2);
src_pos = [0.32, 0.22, 0.15]; % adjust to actual source you picked
direct_delay = norm(rx_pos - src_pos) / 343;
plot(t, h, 'b'); hold on;
xline(direct_delay, 'g--', 'direct path');
xlim([0, 0.1]);
xlabel('Time [s]'); ylabel('Amplitude');
title(sprintf('First 100 ms — direct path expected at %.1f ms', direct_delay*1000));
grid on;

% magnitude spectrum check (should match RTF magnitude)
subplot(3,1,3);
H_check = fft(h);
plot(freqs, abs(RTF), 'b', 'LineWidth', 1.5); hold on;
plot(freqs, abs(H_check(1:K)), 'r--');
xlabel('Frequency [Hz]'); ylabel('|H(f)|');
title('Spectrum sanity check (blue: original RTF, red dashed: FFT of IFFT)');
legend('original |RTF|', '|FFT(h)|');
grid on; xlim([0 500]);