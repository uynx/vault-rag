{ pkgs, lib, ... }:
{
  packages = [ pkgs.uv pkgs.gh pkgs.git ];
  languages.python = {
    enable = true;
    package = pkgs.python312;
  };
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
  env.UV_PYTHON_DOWNLOADS = "never";
  env.UV_PYTHON = "${pkgs.python312}/bin/python";
}
