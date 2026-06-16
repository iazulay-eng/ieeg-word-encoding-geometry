import numpy as np
import scipy
import matplotlib.pyplot as plt # necessary for debug
import time


def my_hilbert(x, block_size=1024, overlap=0.5, axis=-1):
    """Compute the Hilbert transform along a specified axis in smaller blocks."""
    x = np.asarray(x)
    N = x.shape[axis]
    step = int(block_size * (1 - overlap))  # Step size for moving blocks
    num_blocks = (N - block_size) // step + 1  # Number of blocks
    analytic_signal = np.zeros_like(x, dtype=np.complex64)
    window = np.bartlett(block_size)  # Smooth transition at edges

    for i in range(num_blocks):
        start = i * step
        end = min(start + block_size, N)

        block_slice = [slice(None)] * x.ndim
        block_slice[axis] = slice(start, end)
        block = x[tuple(block_slice)]

        L = block.shape[axis]

        # Compute FFT along the specified axis
        Xf = np.fft.fft(block, n=L, axis=axis)

        # Construct Hilbert transform filter
        H = np.zeros(L, dtype=np.float32)
        H[0] = 1
        if L % 2 == 0:
            H[L // 2] = 1  # Nyquist frequency
        H[1:L // 2] = 2

        # Apply the Hilbert filter
        transformed_block = np.fft.ifft(Xf * H, n=L, axis=axis)

        # Apply the window
        transformed_block *= window

        # Overlap-add reconstruction
        analytic_signal[tuple(block_slice)] += transformed_block

    return analytic_signal


def split_to_subbands1(data, sub_centers=[70, 90, 110, 130, 150], subs_bw=20, fs=500, show_dbg=False, dbg_markers=None, normalize=False):

    num_chans = data.shape[0]

    # filter design
    desired = (0, 0, 1, 1, 0, 0) # BPF
    tband = subs_bw / 8 # size of transition band
    filters = []
    ntaps = 77
    ntaps = int(3 * fs / subs_bw)
    ntaps = ntaps + (ntaps % 2) + 1


    plot_responces = False
    if plot_responces:
        fig, ax = plt.subplots(1, len(sub_centers))

    for i_center, center in enumerate(sub_centers):

        bands = (0, center - (subs_bw + tband) / 2, center - (subs_bw - tband) / 2, center + (subs_bw - tband) / 2, center + (subs_bw + tband) / 2, fs / 2)
        fir_firls = scipy.signal.firls(ntaps, bands, desired, fs=fs)
        fir_remez = scipy.signal.remez(ntaps, bands, desired[::2], fs=fs)
        fir_firwin2 = scipy.signal.firwin2(ntaps, bands, desired, fs=fs)

        if plot_responces:
            for b in [fir_firls, fir_remez, fir_firwin2]:
                freq, response = scipy.signal.freqz(b, 1, fs=fs)
                ax[i_center].plot(freq, 20 * np.log10(np.maximum(np.abs(response), 1e-5)))
            ax[i_center].grid(True)
            ax[i_center].legend(['firls', 'remez', 'firwin2', 'firwin'])

        # using the commented-out visualizing code, I have chossen firls
        filters.append(fir_firls)

    if plot_responces:
        plt.show()

    nbands = len(sub_centers)
    nchans, nsamps = data.shape
    output_data = np.zeros((nchans, nbands, nsamps))
    for i_band, filter in enumerate(filters):
        output_data[:, i_band, :] = scipy.signal.filtfilt(filter, 1, data)

    #h = scipy.signal.hilbert(output_data, axis=-1)
    #
    # if output_data.shape[-1] > 3000000:
    #     rc = 'signal longer that 1h:50m'
    #     return rc, None, None
    h = np.zeros(output_data.shape, dtype=np.complex64)
    for i1 in range(output_data.shape[0]):
        for i2 in range(output_data.shape[1]):
            h[i1, i2] = my_hilbert(output_data[i1, i2], block_size=4096)
            # plt.plot(np.real(h[i1, i2]), 'b')
            # plt.plot(np.imag(h[i1, i2]), 'r')
            # h[i1, i2] = scipy.signal.hilbert(output_data[i1, i2])
            # plt.plot(np.real(h[i1, i2]), 'c')
            # plt.plot(np.imag(h[i1, i2]), 'm')
            # plt.show()
    #
    p = np.real(h * np.conj(h))

    # 10ms averaging window
    k_len = int(fs * 10 / 1000)
    k = np.ones(k_len) / k_len
    for i1 in range(p.shape[0]):
        for i2 in range(p.shape[1]):
            p[i1, i2] = np.convolve(p[i1, i2], k, mode='same')
            if normalize:
                p[i1, i2] = p[i1, i2] / p[i1, i2].mean()

    return 0, output_data, p



def plot_signals_by_chan_and_band(signal1, signal2=None, avgsignal=None, plot_only_avg=False,
                                  band_centers=None, bw=0, fs=500, markers=None, chan_names=None, tscope=[-1, 2], ovrd_ylim=None):

    nchans, nbands, nsamps = signal1.shape
    if signal2 is not None:
        assert signal1.shape == signal2.shape

    # set the time scope relative to marker
    scope_smps = np.arange(start=tscope[0] * fs, stop=tscope[1] * fs).astype(int)
    scope_time = scope_smps / fs

    #fig, ax = plt.subplots(nchans, nbands, sharex=True, sharey=True, figsize=(20, 10))
    ncols = nbands + (avgsignal is not None) if not plot_only_avg else 1
    fig, ax = plt.subplots(5, ncols, sharex=True, sharey=True, figsize=(20, 10))
    ax = ax.reshape(5, ncols)

    smin, smax, qmin, qmax = np.inf, -np.inf, np.inf, -np.inf

    for i_marker, marker in enumerate(markers):
        smps = marker + scope_smps
        for i_chan in range(min(nchans, 5)):
            ax[i_chan, 0].set_ylabel(chan_names[i_chan])
            for i_band in range(nbands):
                if not plot_only_avg:
                    ax[-1, i_band].set_xlabel(str(band_centers[i_band] - bw/2) + '  -  ' + str(band_centers[i_band] + bw/2))
                    ax[i_chan, i_band].plot(scope_time, signal1[i_chan, i_band, smps[0]:smps[-1]+1])
                    smin = min(smin, signal1[i_chan, i_band, smps[0]:smps[-1]+1].min())
                    qmin = min(qmin, np.quantile(signal1[i_chan, i_band, smps[0]:smps[-1]+1], 0.01))
                    smax = max(smax, signal1[i_chan, i_band, smps[0]:smps[-1]+1].max())
                    qmax = max(qmax, np.quantile(signal1[i_chan, i_band, smps[0]:smps[-1]+1], 0.99))
                    if signal2 is not None:
                        ax[i_chan, i_band].plot(scope_time, signal2[i_chan, i_band, smps[0]:smps[-1]+1])
                    ax[i_chan, i_band].plot([0, 0], [-10, 10], c='k')
                    #ax[i_chan, i_band].set_xlim(tscope)
                    ax[i_chan, i_band].grid(True)
            if avgsignal is not None:
                assert type(avgsignal) == type([])
                ax[i_chan, -1].plot(scope_time, avgsignal[0][i_chan, smps[0]:smps[-1] + 1], c='k')
                ax[i_chan, -1].grid(True)
                if len(avgsignal) > 1:
                    ax[i_chan, -1].plot(scope_time, avgsignal[0][i_chan, smps[0]:smps[-1] + 1]  - avgsignal[1][i_chan, smps[0]:smps[-1] + 1], c='b', linestyle=':')
                    ax[i_chan, -1].plot(scope_time, avgsignal[0][i_chan, smps[0]:smps[-1] + 1]  + avgsignal[1][i_chan, smps[0]:smps[-1] + 1], c='b', linestyle=':')

        # set plot span
        qspan = qmax - qmin
        smin = max(smin, qmin - 0.2 * qspan) - 0.02 * qspan
        smax = min(smax, qmax + 0.2 * qspan) + 0.02 * qspan
        for i_chan in range(min(nchans, 5)):
            ax[i_chan, 0].set_ylabel(chan_names[i_chan])
            if ovrd_ylim is None:
                ax[i_chan, 0].set_ylim([smin, smax])
            else:
                ax[i_chan, 0].set_ylim([min(ovrd_ylim), max(ovrd_ylim)])

    return fig


def calc_HFB(data, sub_centers=[70, 90, 110, 130, 150], subs_bw=20, fs=500, tscope=[-0.6, 1.6], dbg_markers=None, chan_names=None,
             plot_prefix=None, gen_plots=True, precalc=None):

    if precalc is None:
        rc, x, p = split_to_subbands1(data, sub_centers=sub_centers, subs_bw=subs_bw, fs=fs, show_dbg=False, dbg_markers=dbg_markers)
        if rc:
            return rc, None, None, None, None
    else:
        x, p = precalc[0], precalc[1]

    num_markers = len(dbg_markers)
    ave_tscope = tscope#[-0.6, 1.6]#[-2, 10.5]
    ave_scope_smps = np.arange(start=ave_tscope[0] * fs, stop=ave_tscope[1] * fs).astype(int)
    ave_scope_time = ave_scope_smps / fs
    norm_meas_mask = (ave_scope_time > -0.5) * (ave_scope_time < 0)
    num_markers_to_process = 300#26
    # band_factors = np.array(sub_centers) / sub_centers[0]
    # for i_band in range(len(band_factors)):
    #     x[:, i_band] *= band_factors[i_band]
    #     p[:, i_band] *= band_factors[i_band]

    p_ave = np.zeros((p.shape[0], p.shape[1], ave_scope_smps.size)) # average over epochs
    grand_p_ave = np.zeros((p.shape[0], ave_scope_smps.size)) # average over bands and epochs
    grand_p_ave_sqr = np.zeros(grand_p_ave.shape) # for calculating standad deviation

    for i_marker, marker in enumerate(dbg_markers[:num_markers_to_process]):

        # normalizing by precursor
        smps = marker + ave_scope_smps
        normalize_every_epoch = True
        if normalize_every_epoch:
            # epoch_x = np.copy(x[:, :, smps[0] : smps[-1] + 1])
            # epoch_p = np.copy(p[:, :, smps[0] : smps[-1] + 1])
            epoch_x = x[:, :, smps[0] : smps[-1] + 1]
            epoch_p = p[:, :, smps[0] : smps[-1] + 1]
            if epoch_p.shape[-1] != norm_meas_mask.size:
                print('here')
            ave_p = epoch_p[:, :, norm_meas_mask[:epoch_p.shape[-1]]].mean(axis=-1)
            for i_chan in range(ave_p.shape[0]):
                for i_band in range(ave_p.shape[1]):
                    epoch_x[i_chan, i_band] /= np.sqrt(ave_p[i_chan, i_band])
                    epoch_p[i_chan, i_band] /= ave_p[i_chan, i_band]

        # PATCH
        p_bandavg = p.mean(axis=1)
        grand_p_ave += p_bandavg[:, smps[0] : smps[-1] + 1] * (1 / len(dbg_markers))
        grand_p_ave_sqr += (p_bandavg[:, smps[0] : smps[-1] + 1] ** 2) * (1 / len(dbg_markers))
        #

        # averaging
        smps = marker + ave_scope_smps
        p_ave += p[:, :, smps[0] : smps[-1] + 1] * (1 / len(dbg_markers))
        #p_ave += epoch_p * (1 / len(dbg_markers))

        # plotting non-averaged signal
        #print('going to create fig for marker', i_marker)
        if gen_plots:
            fig = plot_signals_by_chan_and_band(x, signal2=np.sqrt(p), avgsignal=[p_bandavg], band_centers=sub_centers, bw=20, fs=500, markers=[marker], chan_names=chan_names, tscope=tscope)
            fig.suptitle('marker ({})  at {:7.2f}'.format(i_marker + 1, marker / fs))
            plt.show()
            if plot_prefix is not None:
                fname = plot_prefix + 'marker_' + str(i_marker + 1)
                fig.savefig(fname)
                #plt.show()
                plt.close()
                plt.clf()
                print('saved:', fname)

    # final processing
    grand_p_ave_var = grand_p_ave_sqr - grand_p_ave ** 2
    sem = np.sqrt(np.maximum(grand_p_ave, 0) / len(dbg_markers))

    if True: #not normalize_every_epoch:
        for i_chan in range(p_ave.shape[0]):
            for i_band in range(p_ave.shape[1]):
                nf = p_ave[i_chan, i_band, norm_meas_mask].mean(axis=-1) + 1e-18
                p_ave[i_chan, i_band] /= nf
                sem /= nf

    if gen_plots:
        fig = plot_signals_by_chan_and_band(np.sqrt(p_ave), avgsignal=[grand_p_ave, sem], plot_only_avg=True,
                                            band_centers=sub_centers, bw=20, fs=500, markers=[np.argmin(np.abs(ave_scope_smps)).squeeze()],
                                            chan_names=chan_names, tscope=tscope, ovrd_ylim=[0.8, 2.2])
        fig.suptitle('avg. over {} events'.format(len(dbg_markers)))
        plt.show()
        if plot_prefix is not None:
            fname = plot_prefix + 'average'
            fig.savefig(fname)
            print('saved:', fname)

        plt.show(block=False)
        plt.pause(0.1)

    return 0, np.sqrt(p_ave), grand_p_ave, sem, [x, p]
    #return np.sqrt(p_ave), np.sqrt(p_ave).mean(axis=1), sem, [x, p]



