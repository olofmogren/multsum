#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, time, getopt, select, threading, socket

from gensim.models import word2vec
from multiprocessing import Process
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

def init(w2v_vector_file=W2V_VECTOR_FILE):
  global wordmodel
  print( 'Initializing. Python version: '+str(sys.version_info.major)+'.'+str(sys.version_info.minor)+'.'+str(sys.version_info.micro))

  print( 'Initializing word model...')
  sys.stdout.flush()
  t = time.time()
  try:
    wordmodel = word2vec.Word2Vec.load_word2vec_format(W2V_VECTOR_FILE, binary=True)
  except IOError, e:
    print 'Error reading '+w2v_vector_file+'. Please provide a working w2v binary file.'
    #wordmodel = None
  print( 'done. %.3f secs'%(time.time()-t))
  sys.stdout.flush()


def run_backend(replace=False, exit=False, w2v_vector_file=W2V_VECTOR_FILE, temp_startup_server=False):
  repr_req_count = 0

  send_exit_command_after_init = False

  if not temp_startup_server:
    try:
      print('Connecting to backend.')
      conn = Client(BACKEND_ADDRESS, family=BACKEND_CONNECTION_FAMILY, authkey=BACKEND_PASSWORD)
      if not replace and not exit:
        conn.send({'command': 'CAPABILITIES'})
        msg = conn.recv()
        if msg['status'] == 'OK' and (msg['value'] == 'FULL' or msg['value'] == 'W2V_ONLY'):
          print('There is already an existing daemon with all neccessary capacities. Exiting.')
          sys.exit()
        else:
          print msg
          print('Found a running backend without required capabilities. Telling it to exit after init.')
          send_exit_command_after_init = True
      else: 
        print('Found a running backend. Telling it to exit after init.')
        send_exit_command_after_init = True
    except Exception, e:
      print(str(e))
      print(str(traceback.format_exc()))
      print('Did not manage to find a backend to kill.')

    if exit:
      print('Exiting.')
      sys.exit()

    if not send_exit_command_after_init:
      # The temp_startup_serer will take care of requests until startup (init()) is complete.
      p = Process(target=run_backend, kwargs={'temp_startup_server': True})
      p.start()
    
    init(w2v_vector_file=w2v_vector_file)
    
    # either temp_startup_server backend, or a previously running backend.
    print('Connecting to temp_startup_server (or previously running) backend.')
    conn = Client(BACKEND_ADDRESS, family=BACKEND_CONNECTION_FAMILY, authkey=BACKEND_PASSWORD)
    conn.send({'command': 'EXIT_NOW'})
    conn.close()
    conn = None
    if not send_exit_command_after_init:
      print('Waiting 1 sec for temp_startup_server backend to clean up.')
      time.sleep(1)
    else:
      print('Waiting 30 sec for previously running backend to clean up.')
      time.sleep(30)
      
    if not send_exit_command_after_init:
      p.terminate()
      print('temp_startup_server backend is now terminated.')
  else:
    print('temp_startup_server running.')

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
          if temp_startup_server and msg['command'] != 'EXIT_NOW' \
                                 and msg['command'] != 'CLOSE' \
                                 and msg['command'] != 'CAPABILITIES':
            conn.send({'status': 'FAIL', 'value': 'Backend is initializing.'})
            continue
          elif msg['command'] == 'wordmodel':
            repr_req_count = repr_req_count+1
            if not wordmodel:
              conn.send({'status': 'FAIL', 'value': '\'Looking for term: '+msg['term']+'\': wordmodel does not seem to be inizialized. Was the vector file not found?'})
            #print msg['command']+': '+msg['term']
            else:
              if msg['term'] in wordmodel:
                conn.send({'status':'OK', 'value': wordmodel[msg['term']]})
              else:
                conn.send({'status': 'FAIL', 'value': '\''+msg['term']+'\': No such term found in '+msg['command']+'.'})
          elif msg['command'] == 'PING':
            if not wordmodel:
              conn.send({'status': 'FAIL', 'value': '\'Got PING. But wordmodel does not seem to be inizialized. Was the vector file not found?'})
            else:
              conn.send({'status': 'OK', 'value': 'PONG'})
          elif msg['command'] == 'CAPABILITIES':
            conn.send({'status': 'OK', 'value': 'W2V_ONLY'})
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

            sys.exit()
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
    -e, --exit:    tell a running daemon to exit.
  '''
  replace = False
  exit = False
  # parse command line options
  try:
    opts, args = getopt.getopt(sys.argv[1:], "r:h:e", ["replace", "help", "exit"])
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
    if o in ("-e", "--exit"):
      print( 'Will try to tell a running backend to exit.')
      exit = True
  run_backend(replace=replace, exit=exit)
