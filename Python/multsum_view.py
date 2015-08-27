#!/usr/bin/python

import sys


def run_browser():
  from gi.repository import Gtk, Gdk, WebKit
  Gtk.init(sys.argv)
  
  #self.ebview.go_back()
  #self.webview.goforward()

  win = Gtk.Window()
  scrolled_window = Gtk.ScrolledWindow()
  webview = WebKit.WebView()
  scrolled_window.add(webview)
  win.add(scrolled_window)
  win.set_title('MULTSUM')

  win.connect("delete-event", Gtk.main_quit)
  #self.webview.show()
  #scrolled_window.show()
  win.show_all()
  webview.load_uri('http://localhost:9191/')
  win.resize(700, 700)

  #browser = Browser()
  Gtk.main()


if __name__ == "__main__":
  run_browser()

