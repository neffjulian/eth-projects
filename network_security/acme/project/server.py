from functools import partial
import ssl
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler

from dnslib import RR, QTYPE, A
from dnslib.server import BaseResolver, DNSServer

class DNSResolver(BaseResolver):
    def __init__(self, ip) -> None:
        super().__init__()
        self.ip = ip
        self.rrs = []

    def resolve(self, request, handler):
        reply = request.reply()
        for rr in self.rrs:
            if(rr[0].get_rname() == request.q.qname and rr[0].rtype == request.q.qtype):
                reply.add_answer(rr[0])

        if(request.q.qtype == QTYPE.A and not request.rr):
            answer = RR(
                rname=request.q.qname,
                rtype=request.q.qtype,
                ttl=60,
                rdata=A(self.ip)
            )
            reply.add_answer(answer)
        return reply
    
    def add_rr(self, zone):
        self.rrs.append(RR.fromZone(zone))

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

class ShutdownHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/shutdown':
            self.server.shutdown()
        self.send_response(200)
        self.end_headers()

def run_dns_server(address): 
    resolver = DNSResolver(address)
    server = DNSServer(resolver, address, port=10053)
    server.resolver = resolver
    t = Thread(target=server.start)
    t.start()
    return server

def run_certificate_server(keyfile="client/.key", certfile="client/.crt"):
    server = HTTPServer(('', 5001), Handler)
    server.socket = ssl.wrap_socket(server.socket, keyfile, certfile)
    t = Thread(target=server.serve_forever)
    t.start()
    return server

def run_challenge_server():
    Handler = partial(SimpleHTTPRequestHandler, directory = "client")
    server = HTTPServer(('', 5002), Handler)
    t = Thread(target=server.serve_forever)
    t.start()
    return server

def run_shutdown_server():
    server = HTTPServer(('', 5003), ShutdownHandler)
    t = Thread(target=server.serve_forever)
    t.start()
    return server

def stop_server(server):
    server.keep_running = False
    server.shutdown()
    server.server_close()
