''' Send email to each name in column 1 of a file
    at corresponding time in column 2.
'''
import sys
from string import Template
import datetime as dt
import pandas as pd
import smtplib
from apscheduler.schedulers.blocking import BlockingScheduler
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# Prompt to enter email configuration parameters.
HOST = input('Enter your email host name: ')
PORT = input('Enter your email port number: ')
MY_ADDRESS = input('Enter your email address: ')
PASSWORD = input('Enter your email password: ')


def connect_to_server(host, port, address, password):
    # Set up the SMTP server
    SERVER = smtplib.SMTP()
    SERVER.connect(host, port)
    SERVER.starttls()
    SERVER.login(address, password)
    return SERVER


def read_message_template(filename):
    ''' Read the template of the email message.

        Parameters
        ----------
        filename: str,
            file name of the email message.

        Returns
        ----------
        Template: Template object
    '''
    with open(filename, 'r', encoding='utf-8') as template_file:
        template_file_content = template_file.read()
    return Template(template_file_content)


def send_email(name, message_template):
    ''' Send email message to the name specified.

        Parameters
        ----------
        name: str,
            name of people who would receive the email.
        message_template: Template object,
            message template
    '''
    # Create a message
    msg = MIMEMultipart()
    # Add in the actual person name to the message template
    message = message_template.substitute(PERSON_NAME=name)
    # Set up the parameters of the message
    # Email address from whom to send the message.
    msg['From'] = MY_ADDRESS
    # Email address to whom to send the message.
    msg['To'] = '{}@hku.hk'.format(name)
    # Email message subject.
    msg['Subject'] = "Meeting"
    # Add in the message body
    msg.attach(MIMEText(message, 'plain'))
    # Connect to server every time before sending the email
    # to avoid disconnection.
    SERVER = connect_to_server(HOST, PORT, MY_ADDRESS, PASSWORD)
    # Send the message via the server.
    SERVER.send_message(msg)
    # Terminate the SMTP session and close the connection
    SERVER.quit()
    return


def main():
    # Read email recipients and corresponding sending time.
    name_time = pd.read_excel("Email.csv", header=None, names=['name', 'time'])
    message_template = read_message_template('message.txt')
    # Instantiate BlockingScheduler.
    sched = BlockingScheduler()
    # Iterate over the name and time, send the email:
    for index, row in name_time.iterrows():
        name_, time_ = row['name'], row['time']
        # Add jobs to the scheduler.
        sched.add_job(send_email, 'date',
                      run_date=dt.datetime.combine(dt.date.today(), time_),
                      args=[name_, message_template])
    # Start the scheduler.
    sched.start()
    return 0


if __name__ == '__main__':
    STATUS = main()
    sys.exit(STATUS)
