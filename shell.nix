{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = with pkgs; [
    vulkan-loader
  ];

  LD_LIBRARY_PATH = "\$\{LD_LIBRARY_PATH\}:${ pkgs.lib.makeLibraryPath [ pkgs.vulkan-loader ] }";
}
