import os
import argparse
from jws import RSAKey
import server
from client import Client


def main(challenge_type, dir_url: str, record, domains, revoke):
    os.makedirs("client/.well-known/acme-challenge", exist_ok=True)
    client = Client(dir_url)
    shutdown_server = server.run_shutdown_server()
    challenge_server = server.run_challenge_server()
    dns_server = server.run_dns_server(record)
    client.create_account()
    client.submit_order(domains, challenge_type)
    client.prove_control(dns_server)

    key = RSAKey()
    client.finalize_order(key)
    certificate = client.download_certificate()
    client.write_certificate_and_key(certificate, key)

    certificate_server = server.run_certificate_server()
    if revoke:
        client.revoke_certificate(certificate)
    
    server.stop_server(shutdown_server)
    server.stop_server(challenge_server)
    server.stop_server(certificate_server)
    dns_server.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("type")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--domain", required=True, action='append', nargs='+')
    parser.add_argument("--revoke", action="store_true")

    arguments = parser.parse_args()
    main(arguments.type, arguments.dir, arguments.record, arguments.domain, arguments.revoke)
