#include <iostream>

#include "controller.hh"
#include "timestamp.hh"

using namespace std;

/* Default constructor */
Controller::Controller( const bool debug )
  : debug_( debug )
{}

int the_window_size = 1;
int ssthresh = 500;
int received = 0;
int timeout = 0;
int nack = 0;
unsigned int last_send;
unsigned int last_ack;

/* Get current window size, in datagrams */
unsigned int Controller::window_size() {
  if (received >= the_window_size) {
    received = 0;
    if (the_window_size <= ssthresh >> 1) {
      the_window_size = the_window_size << 1;
    } else if (the_window_size < ssthresh) {
      the_window_size += 2;
    } else {
      the_window_size += 1;
    }
    //cerr << timeout << "   " << nack << "   " << last_send - last_ack << "   " << the_window_size << endl;
  }
  
  if ( debug_ ) {
    cerr << "At time " << timestamp_ms()
	 << " window size is " << the_window_size << endl;
  }

  return the_window_size;
}

/* A datagram was sent */
void Controller::datagram_was_sent( const uint64_t sequence_number,
				    /* of the sent datagram */
				    const uint64_t send_timestamp,
                                    /* in milliseconds */
				    const bool after_timeout
				    /* datagram was sent because of a timeout */ )
{
  if (after_timeout) {
      ssthresh = the_window_size >> 1;
      the_window_size = 1;
      timeout += 1;
  }
  last_send = sequence_number;
  if ( debug_ ) {
    cerr << "At time " << send_timestamp
	 << " sent datagram " << sequence_number << " (timeout = " << after_timeout << ")\n";
  }
}

/* An ack was received */
void Controller::ack_received( const uint64_t sequence_number_acked,
			       /* what sequence number was acknowledged */
			       const uint64_t send_timestamp_acked,
			       /* when the acknowledged datagram was sent (sender's clock) */
			       const uint64_t recv_timestamp_acked,
			       /* when the acknowledged datagram was received (receiver's clock)*/
			       const uint64_t timestamp_ack_received )
                               /* when the ack was received (by sender) */
{
  /* Default: take no action */
  received += 1;
  last_ack = sequence_number_acked;

  if (timestamp_ack_received - send_timestamp_acked > timeout_ms()){
      ssthresh = (the_window_size >> 1);
      the_window_size = the_window_size >> 1;
      nack += 1;
    }

  if ( debug_ ) {
    cerr << "At time " << timestamp_ack_received
	 << " received ack for datagram " << sequence_number_acked
	 << " (send @ time " << send_timestamp_acked
	 << ", received @ time " << recv_timestamp_acked << " by receiver's clock)"
	 << endl;
  }
}

/* How long to wait (in milliseconds) if there are no acks
   before sending one more datagram */
unsigned int Controller::timeout_ms()
{
  return 250; /* timeout of one second */
}
