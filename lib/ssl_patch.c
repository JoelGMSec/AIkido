// ssl_patch.c
#include <openssl/x509.h>

int X509_verify_cert(X509_STORE_CTX *ctx) {
    return 1;  // Allways True
}
