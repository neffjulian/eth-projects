import requests
import json
import hashlib
import time

import helper
from jws import JWS, RSAKey
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization

class Client:
    def __init__(self, dir) -> None:
        self.key = RSAKey()
        self.session = requests.Session()
        self.session.verify = "pebble.minica.pem"
        self.dir = self.session.get(dir).json()
        self.nonces = []
        self.challenges = []

    def create_account(self):
        response = self.get_request(
            url = self.dir['newAccount'],
            payload = {'termsOfServiceAgreed': True}
        )
        self.key.kid = response.headers['Location']

    def submit_order(self, domains: list, type: str):
        values = [domain for subdomains in domains for domain in subdomains]
        response = self.get_request(
            url = self.dir['newOrder'],
            payload = {
                'identifiers': [{
                    'type': 'dns',
                    'value': value
                } for value in values]
            }
        )
        self.url = response.headers.get('Location')
        self.order = response.json()
        challenge_type = {"dns01": "dns-01", "http01": "http-01"}[type]
        for i in enumerate(self.order['authorizations']): # i = (i, auth[i])
            res_auth = self.get_request(
                url = i[1],
                payload = b''
            ).json()
            for challenge in res_auth['challenges']:
                if challenge['type'] == challenge_type:
                    self.challenges.append({
                        'identifier': res_auth['identifier']['value'],
                        'type': challenge_type,
                        'url': challenge['url'],
                        'token': challenge['token'],
                    })

    def prove_control(self, dnsServer):
        for challenge in self.challenges:
            account_key = json.dumps(
                self.key.jwk(), 
                sort_keys=True, 
                separators=(",",":")
            ).encode('utf-8')
            thumbprint = self.get_sha_digest(account_key)
            key_authorization = helper.encode_utf_8(challenge['token'] + "." + thumbprint)
            if challenge['type'] == "http-01":
                location = "client/.well-known/acme-challenge/" + challenge['token']
                with open(location, "wb") as file:
                    file.write(key_authorization)
            elif challenge['type'] == "dns-01":
                sha_digest = self.get_sha_digest(key_authorization)
                name = challenge['identifier']
                zone = f'_acme-challenge.{name}. 300 TXT "{sha_digest}"'
                dnsServer.resolver.add_rr(zone)
            response = self.get_request(
                url = challenge['url'],
                payload = b'{}'
            ).json()

    def get_sha_digest(self, account_key):
        thumbprint = helper.decode_utf_8(
            helper.encode_base_64(
                hashlib.sha256(account_key).digest()
            )
        )
        return thumbprint

    def finalize_order(self, key:RSAKey):
        nameAttribute = x509.NameAttribute(
            oid = x509.NameOID.COMMON_NAME, 
            value = self.order['identifiers'][0]['value'])
        general_names = []
        for identifier in self.order['identifiers']:
            general_names.append(x509.DNSName(identifier['value']))
        csr = x509.CertificateSigningRequestBuilder()
        csr = csr.subject_name(name = x509.Name([nameAttribute]))
        csr = csr.add_extension(
            extval =  x509.SubjectAlternativeName(
                general_names = general_names
            ), 
            critical = False
        )
        csr = csr.sign(
            private_key = key.private_key, 
            algorithm = hashes.SHA256(), 
            backend = default_backend())
        csr = csr.public_bytes(serialization.Encoding.DER)
        csr = helper.decode_utf_8(
            helper.encode_base_64(
                msg = csr
            )
        )
        time.sleep(5)
        response = self.get_request(
            url = self.order['finalize'], 
            payload = helper.encode_utf_8(
                msg = json.dumps({
                    'csr': csr
                })
            )).json()

        timeout = time.time() + 60
        while(time.time() < timeout):
            response = self.get_request(self.url, b'').json()
            if(response['status'] == 'valid'):
                self.order['chalz'] =  response['certificate']
                return
            time.sleep(2)
        raise Exception("TIMEOUT")

    def get_request(self, url: str, payload={}, headers: dict = {}, key = None):
        if key == None:
            key = self.key
        jws = JWS(key)
        if self.nonces:
            nonce = self.nonces.pop()
        else:
            nonce = self.session.head(
                url = self.dir['newNonce']
            ).headers['Replay-Nonce']
        header = {
            'url': url,
            'nonce': nonce,
            'alg': 'RS256'
        }
        headers.update({
            'Content-Type': 'application/jose+json'
        })
        response = self.session.post(
            url = url,
            headers = headers,
            data = jws.create(
                header = header,
                payload = payload
            )
        )
        if response.headers['Replay-Nonce']:
            self.nonces.append(response.headers['Replay-Nonce'])
        return response

    def download_certificate(self):
        response = self.get_request(
            url = self.order['chalz'],
            payload = b'',
            headers = {
                'Accept': "application/pem-certificate-chain"
            }
        )
        return response.content
        
    def write_certificate_and_key(self, certificate, key, location="client/"):
        with open(location + ".crt", "wb") as certloc:
            certloc.write(certificate)
        pem = key.save_pem()
        with open(location + ".key", "wb") as keyloc:
            keyloc.write(pem)

    def revoke_certificate(self, certificate):
        certificate = self.key.load_pem(certificate)
        payload = helper.encode_utf_8(
            msg = json.dumps({
                'Certificate': helper.decode_utf_8(
                    bts = helper.encode_base_64(certificate)
                )
            })
        )
        response = self.get_request(
            url = self.dir["revokeCert"],
            payload = payload
        ).json()
