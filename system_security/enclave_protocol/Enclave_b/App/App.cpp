#include <stdio.h>
#include <cstdio>
#include <string.h>
#include <assert.h>
# include <unistd.h>
# include <pwd.h>
# define MAX_PATH FILENAME_MAX

#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h"
#include "sgx_tcrypto.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#define SGX_ECP256_KEY_SIZE  32


/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 1;

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
} sgx_errlist_t;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
    {
        SGX_ERROR_INVALID_PARAMETER,
        "Invalid parameter.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_MEMORY,
        "Out of memory.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_LOST,
        "Power transition occurred.",
        "Please refer to the sample \"PowerTransition\" for details."
    },
    {
        SGX_ERROR_INVALID_ENCLAVE,
        "Invalid enclave image.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ENCLAVE_ID,
        "Invalid enclave identification.",
        NULL
    },
    {
        SGX_ERROR_INVALID_SIGNATURE,
        "Invalid enclave signature.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_EPC,
        "Out of EPC memory.",
        NULL
    },
    {
        SGX_ERROR_NO_DEVICE,
        "Invalid SGX device.",
        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
    },
    {
        SGX_ERROR_MEMORY_MAP_CONFLICT,
        "Memory map conflicted.",
        NULL
    },
    {
        SGX_ERROR_INVALID_METADATA,
        "Invalid enclave metadata.",
        NULL
    },
    {
        SGX_ERROR_DEVICE_BUSY,
        "SGX device was busy.",
        NULL
    },
    {
        SGX_ERROR_INVALID_VERSION,
        "Enclave version was invalid.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ATTRIBUTE,
        "Enclave was not authorized.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_FILE_ACCESS,
        "Can't open enclave file.",
        NULL
    },
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }

    if (idx == ttl)
    	printf("Error code is 0x%X. Please refer to the \"Intel SGX SDK Developer Reference\" for more details.\n", ret);
}

/* Initialize the enclave:
 *   Call sgx_create_enclave to initialize an enclave instance
 */
int initialize_enclave(void)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;

    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        return -1;
    }
    return 0;
}



/* OCall functions */
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate
     * the input string to prevent buffer overflow.
     */
    printf("%s", str);
}

/***********************************************
// 1. BEGIN: Send and receive public keys in B
***********************************************/

void send_public_key(sgx_ec256_public_t public_key){
  int fd;
  const char *myfifo = "/tmp/myfifo_b";
  mkfifo(myfifo, 0666);
  fd = open(myfifo, O_WRONLY);

  write(fd, public_key.gx, SGX_ECP256_KEY_SIZE);
  write(fd, public_key.gy, SGX_ECP256_KEY_SIZE);

  close(fd);
}

sgx_ec256_public_t receive_public_key(){
  int fd;
  const char *myfifo = "/tmp/myfifo_a";
  mkfifo(myfifo, 0666);
  fd = open(myfifo,O_RDONLY);

  sgx_ec256_public_t public_key;
  read(fd, public_key.gx, SGX_ECP256_KEY_SIZE);
  read(fd, public_key.gy, SGX_ECP256_KEY_SIZE);

  close(fd);

  return public_key;
}

/***********************************************
// 1. END: Send and receive public keys in B
***********************************************/

/***********************************************
// 1. BEGIN: Send and receive PSK in B
***********************************************/


void encrypt_and_sendC(){
  uint8_t C;
  uint8_t IV[16];
  
  sgx_status_t sgx_stat;
  fetch_iv(global_eid, IV);
  
  sgx_status_t send_status;
  // Send PSK_A from A in form of C
  send_status = get_encrypted_message_psk(global_eid, &sgx_stat, &C);

  if (send_status == SGX_SUCCESS)
    printf("Sending PSK_B worked...\n");
  else{
    printf("Sending PSK_B failed...\n");
    print_error_message(send_status);
  }

  /*uint8_t k = 0;
  for(k= 0; k < (sizeof(IV)/ sizeof(IV[0])); k++){
    // printf("%d", ctr_key[k]);
    // printf("\n");
    printf("%d", IV[k]);
  }*/
  
  int fd;
  const char *myfifo = "/tmp/myfifo_pskb";
  mkfifo(myfifo, 0666);
  fd = open(myfifo, O_WRONLY);

  write(fd, &C, 11 * sizeof(uint8_t));
  write(fd, IV, 16 * sizeof(uint8_t));

  close(fd);
}

void receive_and_checkC(){
  int fd;
  const char * myfifo = "/tmp/myfifo_pska";
  mkfifo(myfifo, 0666);
  fd = open(myfifo, O_RDONLY);

  uint8_t C;
  uint8_t IV[16];

  read(fd, &C, 11 * sizeof(uint8_t));
  read(fd, IV, 16 * sizeof(uint8_t));
  
  close(fd);

  sgx_status_t sgx_stat;
  sgx_status_t ret_status = get_decrypted_message_psk(global_eid, &sgx_stat, &C, IV);

  if (ret_status != SGX_SUCCESS) {
    printf("Decrypting the message didn't work...\n");
    print_error_message(ret_status);
  }
  else
    printf("Decrypting message of PSK_A worked...\n");

}

/***********************************************
// 1. END: Send and receive PSK in B
***********************************************/

/***********************************************
// 1. BEGIN: Receive challenge from app A
***********************************************/

void receive_challenge(){
    u_int8_t challenge;
    sgx_status_t sgx_stat;
    sgx_status_t recv_status;
    
    uint8_t IV[16];
    
    int fd;
    const char *myfifo = "/tmp/myfifo_challengea";
    mkfifo(myfifo, 0666);
    fd = open(myfifo, O_RDONLY);

    read(fd, &challenge, 3 * sizeof(uint8_t));
    read(fd, IV, 16 * sizeof(uint8_t));

    close(fd);

    recv_status = receive_challenge(global_eid, &sgx_stat, &challenge, IV);

    if (recv_status == SGX_SUCCESS)
    printf("Receiving challenge with rand worked...\n");
    else{
    printf("Receiving challenge with rand failed...\n");
    print_error_message(recv_status);  
    }
}

/***********************************************
// 1. END: Receive challenge from app A
***********************************************/

/***********************************************
// 1. BEGIN: Send response to app A
***********************************************/

void send_response(){

    uint8_t response[3];
    uint8_t *ptr_response = (uint8_t *) & response;
    uint8_t IV[16];
  
    sgx_status_t sgx_stat;
    fetch_iv(global_eid, IV);
    
    sgx_status_t send_status;
    // Send response in an encrypted form
    send_status = send_response(global_eid, &sgx_stat, ptr_response);

    if (send_status == SGX_SUCCESS)
    printf("Sending response worked...\n");
    else{
    printf("Sending response failed...\n");
    print_error_message(send_status);
    }

    int fd;
    const char *myfifo = "/tmp/myfifo_responseb";
    mkfifo(myfifo, 0666);
    fd = open(myfifo, O_WRONLY);

    write(fd, ptr_response, 3 * sizeof(uint8_t));
    write(fd, IV, 16 * sizeof(uint8_t));

    close(fd);

}

/***********************************************
// 1. END: Send response to app A
***********************************************/

/* Application entry */
int SGX_CDECL main(int argc, char *argv[]) {
    (void)(argc);
    (void)(argv);

    /* Initialize the enclave */
    if(initialize_enclave() < 0){
        printf("FAILED: Impossible initializing enclave, exiting  ...\n");
        return -1;
    }

    printf("Enclave creation successful....\n");

    sgx_ec256_public_t public_key;

    sgx_status_t enclave_status;
    sgx_status_t sgx_status;

    //Initializing ECC parameters inside the enclave
    enclave_status = create_ecc(global_eid, &sgx_status, &public_key);

    if (enclave_status == SGX_SUCCESS && sgx_status == SGX_SUCCESS)
      printf("ECC creation succeeded...\n");
    else {
      printf("ECC creation failed...\n");
      print_error_message(sgx_status);
      print_error_message(enclave_status);
    }

    /***********************************************
    // 1. BEGIN: SEND and RECEIVE public key App_A
    ***********************************************/

    //Receiving public key from App1
    sgx_ec256_public_t appB_public_key;
    appB_public_key = receive_public_key();
    printf("Received public key from AppA...\n");

    //Sending public key from App1
    send_public_key(public_key);
    printf("Sent public key to AppA...\n");

    /***********************************************
    // 1. END: SEND and RECEIVE public key App_A
    ***********************************************/

    //Derive DH shared key and initialize IV on the enclave
    sgx_status = derive_shared_key(global_eid, &enclave_status, &appB_public_key);
    if (enclave_status == SGX_SUCCESS && sgx_status == SGX_SUCCESS)
      printf("Computing shared key and IV worked...\n");
    else {
      printf("Computing shared key and IV didn't work...\n");
      print_error_message(sgx_status);
      print_error_message(enclave_status);
    }

    /***********************************************
    // 1. BEGIN: Encrypted PSK to App_A, from App_B
    ***********************************************/

    // Receive PSK_A from AppA
    receive_and_checkC();
    
    // Send PSK_B to AppA
    encrypt_and_sendC();

    /***********************************************
    // 1. END: Encrypted PSK to App_A, from App_B
    ***********************************************/

    /***********************************************
    // 1. BEGIN: Receive challenge from A
    ***********************************************/

    receive_challenge();

    /***********************************************
    // 1. END: Receive challenge from A
    ***********************************************/

    /***********************************************
    // 1. BEGIN: Send response to app A
    ***********************************************/

    send_response();

    /***********************************************
    // 1. END: Send response to app A
    ***********************************************/



    /* Utilize edger8r attributes */
    edger8r_array_attributes();
    edger8r_pointer_attributes();
    edger8r_type_attributes();
    edger8r_function_attributes();

    /* Utilize trusted libraries */
    ecall_libc_functions();
    ecall_libcxx_functions();
    ecall_thread_functions();

    printSecret(global_eid, &sgx_status);
    if (sgx_status != SGX_SUCCESS) {
        print_error_message(sgx_status);
        return -1;
    }

    /* Destroy the enclave */
    sgx_destroy_enclave(global_eid);

    printf("From App: Enclave destroyed\n");
    return 0;
}