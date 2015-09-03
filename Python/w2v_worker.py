#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, time, getopt, select, threading, socket

from gensim.models import word2vec

from multiprocessing.connection import Listener, Client
from array import array

import traceback

BACKEND_PORT              = 6000
BACKEND_HOST              = 'localhost'
BACKEND_ADDRESS           = (BACKEND_HOST, BACKEND_PORT)
BACKEND_PASSWORD          = 'the secretest password'
BACKEND_CONNECTION_FAMILY = 'AF_INET' #'AF_UNIX' #'AF_INET', 'AF_PIPE'
W2V_VECTOR_FILE = '/home/mogren/tmp/GoogleNews-vectors-negative300.bin'

wordmodel = None

def init():
  global wordmodel
  print( 'Initializing. Python version: '+str(sys.version_info.major)+'.'+str(sys.version_info.minor)+'.'+str(sys.version_info.micro))

  print( 'Initializing word model...')
  sys.stdout.flush()
  t = time.time()
  wordmodel = word2vec.Word2Vec.load_word2vec_format(W2V_VECTOR_FILE, binary=True)
  print( 'done. %.3f secs'%(time.time()-t))
  sys.stdout.flush()


def run_backend(replace=False):
  init()

  if replace:
    try:
      print( 'Trying to replace a running backend.')
      print( 'Connecting to backend.')
      conn = Client(BACKEND_ADDRESS, family=BACKEND_CONNECTION_FAMILY, authkey=BACKEND_PASSWORD)
      conn.send({'command': 'EXIT_NOW'})
      conn.close()
      conn = None
      print( 'Waiting 30 secs for old backend to clean uo.')
      time.sleep(30)
    except:
      print( 'Did not manage to find a backend to kill.')

  connections = {}
  epoll = select.epoll()

  done = False
  while not done:
    try:
      listener = Listener(BACKEND_ADDRESS, family=BACKEND_CONNECTION_FAMILY, authkey=BACKEND_PASSWORD)
      listener_fileno = listener._listener._socket.fileno()
      epoll.register(listener_fileno, select.EPOLLIN)
      done = True
    except socket.error, e:
      print( 'Sleeping a bit, then trying again.')
      time.sleep(30)

  print_secs = int(time.time())
  while True:
    if (int(time.time())-print_secs) > 600:
      # At least 600 secs between printouts.
      print( 'Currently %d clients, %d reps since last print. Last accpt: \'%s\'.'%(len(connections), repr_req_count, str(listener.last_accepted)))
      print_secs = int(time.time())


    events = epoll.poll(1)
    for fileno, event in events:
      try:
        if fileno == listener_fileno:
          print( 'Accepting connection...')
          conn = listener.accept()
          print( 'Connection accepted from: %s'%(str(listener.last_accepted)))
          connections[conn.fileno()] = conn
          epoll.register(conn.fileno(), select.EPOLLIN)
        else:
          conn = connections[fileno]
          msg = conn.recv()
          # do something with msg
          if msg['command'] == 'wordmodel':
            repr_req_count = repr_req_count+1
            #print msg['command']+': '+msg['term']
            if msg['term'] in wordmodel:
              conn.send({'status':'OK', 'value': wordmodel[msg['term']]})
            else:
              conn.send({'status': 'FAIL', 'value': '\''+msg['term']+'\': No such term found in '+msg['command']+'.'})
          elif msg['command'] == 'PING':
            conn.send({'status': 'OK', 'value': 'PONG'})
          elif msg['command'] == 'CLOSE':
            epoll.unregister(conn.fileno())
            del connections[conn.fileno()]
            conn.close()
            break
          elif msg['command'] == 'EXIT_NOW':
            print( 'Received EXIT_NOW! Will exit.')
            #best-effort cleanup:
            try:
              epoll.unregister(listener_fileno)
              listener.close()
              epoll.unregister(conn.fileno())
              del connections[conn.fileno()]
              conn.close()
            except:
              pass
            for c in connections:
              try:
                epoll.unregister(c.fileno())
                connections[c].close()
              except:
                pass

            exit()
            #return
          else:
            conn.send({'status': 'FAIL', 'value': 'Unknown command.'})
            
      except Exception, e:
        print( str(e))
        print( str(traceback.format_exc()))
        print( 'Closing connection.')
        if conn:
          epoll.unregister(conn.fileno())
          del connections[conn.fileno()]
          conn.close()
        break

if __name__ == "__main__":
  '''
    -r, --replace: replace a running daemon.
  '''
  replace = False
  # parse command line options
  try:
    opts, args = getopt.getopt(sys.argv[1:], "r:h", ["replace", "help"])
  except getopt.GetoptError, msg:
    print( str(msg))
    print( "for help use --help")
    sys.exit(2)
  # process options
  for o, a in opts:
    if o in ("-h", "--help"):
      print( __doc__)
      sys.exit(0)
    if o in ("-r", "--replace"):
      print( 'Will try to replace a running backend. First initialize.')
      replace = True
  run_backend(replace)
