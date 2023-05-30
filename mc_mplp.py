class Channels():

    def __init__(self, xs: 'np.array', **kwargs):
        self.xs = xs
        self.channels = dict()
        for k in kwargs:
            self.channels[k] = kwargs[k]

    def __get_channel(self, __name: str) -> 'np.array':
        if __name == 'xs':
            return self.xs
        else:
            return self.channels[__name]

    def __getattr__(self, __name: str) -> 'np.array':
        return self.__get_channel(__name)

    def __getitem__(self, __name: str) -> 'np.array':
        return self.__get_channel(__name)

    @abstractmethod
    def debug_plot(self, outfile: str) -> None:
        pass

    @abstractmethod
    def stack(self) -> 'np.array':
        pass

class Channels2d(Channels):

    def __init__(self, xs: 'np.array', **kwargs):
        super().__init__(xs, **kwargs)
        self.mchannels = dict()
        self._ymax = -1

    def _get_max(self) -> float:
        if self._ymax == -1:
            ymax = -1
            for k in self.channels:
                if (cmax := np.max(self.channels[k])) > ymax:
                    ymax = cmax
            self._ymax = int(np.ceil(ymax))
        return self._ymax

    @abstractmethod
    def build_matrix(self, size: tuple) -> None:
        pass

    def stack(self) -> 'np.array':
        if len(self.mchannels) == 0:
            raise Exception(
                "Matrices are not built yet. Please use build_matrix() before."
            )
        return np.stack([self.mchannels[k] for k in self.mchannels], axis=0)

    def debug_plot(self, outfile="debug.png") -> None:
        fig, axs = plt.subplots(len(self.mchannels),
                                sharex=True,
                                figsize=(7, 2 * len(self.mchannels)))
        for i, k in enumerate(self.mchannels):
            axs[i].imshow(self.mchannels[k])
            axs[i].set_title(k)

        plt.tight_layout()
        fig.savefig(outfile)
        plt.close()


class ChannelsMat2d(Channels2d):

    def _getmat(self, channel: 'np.array', h: int, size: tuple) -> 'np.array':
        w = self.xs.shape[0]
        if np.max(channel) == 0:
            return np.zeros(size).astype(int)

        a = np.zeros((h, w))
        a = a.T
        for i in range(a.shape[0]):
            ii = np.r_[0:int(channel[i])]
            a[i][[ii]] = 1
        a = a.T
        a = np.flip(a, axis=0)
        r = skresize(a, size)
        r = np.rint(r)
        r = (r > 0).astype(int)

        return r

    def build_matrix(self,
                     size: tuple = (200, 1000),
                     _version: int = 3) -> None:
        if not _version in range(3 + 1):
            raise Exception(f"Version {_version} does not exists.")

        _ymax = self._get_max()
        if _version == 3:
            self.mchannels["specific_counts"] = self._getmat(
                self.specific_counts - self.specific_skips, _ymax, size)
            self.mchannels["specific_skips"] = self._getmat(
                self.specific_skips, _ymax, size)
            self.mchannels["nonspecific_counts"] = self._getmat(
                self.nonspecific_counts - self.nonspecific_skips, _ymax, size)
            self.mchannels["nonspecific_skips"] = self._getmat(
                self.nonspecific_skips, _ymax, size)
            self.mchannels["alignment_counts"] = self._getmat(
                self.alignment_counts - self.alignment_skips, _ymax, size)
            self.mchannels["alignment_skips"] = self._getmat(
                self.alignment_skips, _ymax, size)
            self.mchannels["exon"] = self._getmat(
                self.exon_counts, int(np.ceil(np.max(self.exon_counts))), size)
        else:
            for k in self.channels:
                _ys = self.channels[k]
                self.mchannels[k] = self._getmat(
                    _ys,
                    int(np.ceil(np.max(_ys)))
                    if _version == 1 or k == "exon_counts" else _ymax, size)


class ChannelsMplp2d(Channels2d):

    def _getimg(self, size: tuple, *channels: 'np.array',
                _version: int) -> 'np.array':
        _size = (size[1] // 100, size[0] // 100)
        fig = plt.figure(figsize=_size)
        ax = fig.add_axes([0., 0., 1., 1.])
        ax.fill_between(self.xs, channels[0], color="darkgray")
        if len(channels) == 2:
            ax.fill_between(self.xs, channels[1], color="lightgray")
            if _version == 2:
                ax.set_ylim([0, self._get_max()])
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])
        data = np.rint(data)
        plt.close()
        return data

    def build_matrix(self, size: tuple, _version: int = 2) -> None:
        if not _version in range(2 + 1):
            raise Exception(f"Version {_version} does not exists.")

        self.mchannels["specific"] = self._getimg(size,
                                                  self.specific_counts,
                                                  self.specific_skips,
                                                  _version=_version)
        self.mchannels["non_specific"] = self._getimg(size,
                                                      self.nonspecific_counts,
                                                      self.nonspecific_skips,
                                                      _version=_version)
        self.mchannels["alignment"] = self._getimg(size,
                                                   self.alignment_counts,
                                                   self.alignment_skips,
                                                   _version=_version)

        self.mchannels["exon"] = self._getimg(size,
                                              self.exon_counts,
                                              _version=_version)


class ChannelsCAln2d(Channels2d):

    def _color_base(self, base: 'np.array', color: 'np.array',
                    size: tuple) -> 'np.array':
        if np.max(color) == 0:
            return skresize(base, size)
        b = np.copy(base)
        c = np.copy(color)
        c *= (128.0 / c.max())
        c = np.rint(c)
        for i in range(b.shape[1]):
            if c[i] > 0:
                b[:, i][b[:, i] > 0] -= c[i]
        r = skresize(b, size)
        r = np.rint(r)
        return r

    def build_matrix(self,
                     size: tuple = (200, 1000),
                     _version: int = 2) -> None:
        if not _version in range(2 + 1):
            raise Exception(f"Version {_version} does not exists.")

        aln = self.alignment_counts
        w = self.xs.shape[0]
        h = int(np.ceil(np.max(aln)))
        base_aln = np.zeros((h, w))
        base_aln = base_aln.T
        for i in range(base_aln.shape[0]):
            ii = np.r_[0:int(aln[i])]
            base_aln[i][[ii]] = 255
        base_aln = base_aln.T
        base_aln = np.flip(base_aln, axis=0)
        r = skresize(base_aln, size)
        r = np.rint(r)
        r = (r > 0).astype(int)
        self.mchannels["base_aln"] = r
        self.mchannels["alignment_counts"] = self._color_base(
            base_aln, self.alignment_counts - self.alignment_skips, size)
        self.mchannels["alignment_skips"] = self._color_base(
            base_aln, self.alignment_skips, size)
        self.mchannels["specific_counts"] = self._color_base(
            base_aln, self.specific_counts - self.specific_skips, size)
        self.mchannels["specific_skips"] = self._color_base(
            base_aln, self.specific_skips, size)
        if _version == 1:
            self.mchannels["nonspecific_counts"] = self._color_base(
                base_aln, self.nonspecific_counts, size)
        elif _version == 2:
            self.mchannels["nonspecific_counts"] = self._color_base(
                base_aln, self.nonspecific_counts - self.nonspecific_skips,
                size)
            self.mchannels["nonspecific_skips"] = self._color_base(
                base_aln, self.nonspecific_skips, size)
        self.mchannels["exon_counts"] = self._color_base(
            base_aln, self.exon_counts * 128.0, size)