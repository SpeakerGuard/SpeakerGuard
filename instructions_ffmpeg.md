
This is the instruction for installing `ffmpeg` and the external (i.e., not included with ffmpeg) en-/de-coders. 

If you don't use the defenses in `defense/speech_compression.py`, just ignore this instruction.

The instruction targets Ubuntu. For other Linux distributions, you may need to make some modifications, e.g., changing `apt` to the package manager corresponding to your distribution. 

<!-- Also, some operations within this instruction are different depending on whether you own `sudo` privilege or not. -->

## step 1: check whether ffmpeg has been installed. If so, 'uninstall' it
- running `which ffmpeg` in your terminal
- if nothing outputs, go to step 2, since `ffmpeg` has not been installed.

- if it outputs an installed path that is out of your home directory (e.g., `/usr/bin/`), ignore it and go to step 2.
- if it outputs an installed path within your home directory (e.g., `~/.local/bin/`), 'unstall' it by `mv XXX/ffmpeg XXX/ffmpeg-old`

## step 2: check whether `pkg-config` has been installed. If not, install it. `pkg-config` is used by `ffmpeg` to locate external encoders during compilation.
- running `which pkg-config` in your terminal
- if the installed path is printed, go to step 3.
- if nothing outputs, install it: 

  if you own `sudo` privilege: 
  
  `sudo apt-get install pkg-config`

  if you are ordinary user: 
  ```
  wget https://pkgconfig.freedesktop.org/releases/pkg-config-0.29.2.tar.gz

  tar -xzvf pkg-config-0.29.2.tar.gz

  cd pkg-config-0.29.2.tar.gz

  ./configure --prefix=$HOME/.local

  make; make install
  ```

## step 3: download the external en-/de-coders and install them
Download links:
<!-- - [OPUS](https://ftp.osuosl.org/pub/xiph/releases/opus/opus-1.3.1.tar.gz)
- [SPEEX](https://ftp.osuosl.org/pub/xiph/releases/speex/speex-1.2rc2.tar.gz)
- [AMR_nb & AMR_wb_dec](https://sourceforge.net/projects/opencore-amr/files/opencore-amr/opencore-amr-0.1.5.tar.gz/download)
- [AMR_wb_enc](https://sourceforge.net/projects/opencore-amr/files/vo-amrwbenc/vo-amrwbenc-0.1.3.tar.gz/download)
- [MP3](http://nchc.dl.sourceforge.net/project/lame/lame/3.99/lame-3.99.5.tar.gz)
- [AAC](https://sourceforge.net/projects/opencore-amr/files/fdk-aac/fdk-aac-2.0.2.tar.gz/download) -->
- [OPUS](https://ftp.osuosl.org/pub/xiph/releases/opus)
- [SPEEX](https://ftp.osuosl.org/pub/xiph/releases/speex)
- [AMR_nb & AMR_wb_dec](https://sourceforge.net/projects/opencore-amr/files/opencore-amr)
- [AMR_wb_enc](https://sourceforge.net/projects/opencore-amr/files/vo-amrwbenc)
- [MP3](https://sourceforge.net/projects/lame/files/lame/)
- [AAC](https://sourceforge.net/projects/opencore-amr/files/fdk-aac)

Install command:
```
tar -xzvf XXX

cd XXX

./configure --prefix $HOME/.local

make; make install

make check
```
Note: replece XXX with the downloaded file name of en-/de-coders.

## step 4: install ffmpeg

<!-- - `export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig/`
or 

    `export PKG_CONFIG_PATH=/usr/lib/pkgconfig/` or 

    `export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig/` 

    (depending on your installed path of pkg-config) -->
<!-- - `export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig/`

- `git clone https://git.ffmpeg.org/ffmpeg.git`

- `cd ffmpeg`

- `./configure --prefix=$HOME/.local/ffmpeg --disable-x86asm --enable-version3 --enable-libspeex --enable-libopus --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-shared --enable-libmp3lame --enable-libvo_amrwbenc --enable-libfdk-aac --extra-ldflags=-L$HOME/.local/lib --extra-cflags=-I$HOME/.local/include`

- `make; make install` -->

```
export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig/

git clone https://git.ffmpeg.org/ffmpeg.git

cd ffmpeg

./configure --prefix=$HOME/.local/ffmpeg --disable-x86asm --enable-version3 --enable-libspeex --enable-libopus --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-shared --enable-libmp3lame --enable-libvo_amrwbenc --enable-libfdk-aac --extra-ldflags=-L$HOME/.local/lib --extra-cflags=-I$HOME/.local/include

make; make install
```

## step 5: set the environment variable

- `vim ~/.bashrc`

- insert the following commands:
  ```
  export PATH=$HOME/.local/ffmpeg/bin:$PATH

  export LD_LIBRARY_PATH=$HOME/.local/ffmpeg/lib:$HOME/.local/lib:$LD_LIBRARY_PATH
  ```
  
- `source ~/.bashrc`

## step 6: testing your installation
- `cd XXX/SEC4SR`
- `python test_ffmpeg_installation.py 'path of an WAV file'`

If you see "Speech compression is ready. Enjoy It.", then everything is ok.

