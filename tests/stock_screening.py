import yfinance as yf
import json
import os
import bcrypt

class User:
    def __init__(self, user_id, username, password_hash):
        self.user_id = user_id
        self.username = username
        self.password_hash = password_hash
        self.watchlist = {}

    def __str__(self):
        return f"User(user_id={self.user_id}, username='{self.username}')"

    def to_dict(self):
        if isinstance(self.password_hash, bytes):
          password_hash_str = self.password_hash.decode('utf-8')
        else:
           password_hash_str = self.password_hash
        return {
            'user_id': self.user_id,
            'username': self.username,
            'password_hash': password_hash_str,
            'watchlist': self.watchlist
        }

    @classmethod
    def from_dict(cls, user_dict):
        password_hash = user_dict.get('password_hash')
        if isinstance(password_hash, str):
            password_hash = password_hash.encode('utf-8')
        return cls(
            user_dict['user_id'],
            user_dict['username'],
            password_hash
        )

class StockData:
    def __init__(self, symbol, price, pe_ratio, eps, dividend_yield, market_cap, volume, beta):
        self.symbol = symbol
        self.price = price
        self.pe_ratio = pe_ratio
        self.eps = eps
        self.dividend_yield = dividend_yield
        self.market_cap = market_cap
        self.volume = volume
        self.beta = beta
    def __str__(self):
        return f"{self.symbol}: Price={self.price}, P/E={self.pe_ratio}, EPS={self.eps}, Yield={self.dividend_yield}, Cap={self.market_cap}, Volume={self.volume}, Beta={self.beta}"
 
    @classmethod
    def from_yahoo_data(cls, symbol, yahoo_data):
        """Create a StockData object from Yahoo Finance data."""
        return cls(
            symbol=symbol,
            price=yahoo_data.info.get('previousClose', None),
            pe_ratio=yahoo_data.info.get('trailingPE', None),
            eps=yahoo_data.info.get('trailingEps', None),
            dividend_yield=yahoo_data.info.get('dividendYield', None),
            market_cap = yahoo_data.info.get('marketCap', None),
            volume = yahoo_data.info.get('volume', None),
            beta=yahoo_data.info.get('beta', None)
        )

class Filter:
    def __init__(self, criteria, operator):
        self.criteria = criteria  # e.g., {"pe_ratio": 15}
        self.operator = operator # e.g., "less than"
    def apply(self, stocks):
        filtered_stocks = []
        for stock in stocks:
            match = True
            for key, value in self.criteria.items():
                stock_value = getattr(stock, key)
                if stock_value is None:
                    match = False
                    break
                if self.operator == "less than" and stock_value >= value:
                    match = False
                    break
                if self.operator == "greater than" and stock_value <= value:
                    match = False
                    break
                if self.operator == "equal to" and stock_value != value:
                    match = False
                    break
                if self.operator == "not equal to" and stock_value == value:
                    match = False
                    break
                if self.operator == "contains" and str(stock_value).find(value) == -1:
                    match = False
                    break
            if match:
                filtered_stocks.append(stock)
        
        return filtered_stocks

    def __str__(self):
        return f"Filter(criteria={self.criteria}, operator='{self.operator}')"

class StockScreen:
    def __init__(self, screen_id, name, filters = None):
        self.screen_id = screen_id
        self.name = name
        self.filters = filters if filters else []

    def add_filter(self, filter):
        self.filters.append(filter)

    def remove_filter(self, filter):
        self.filters.remove(filter)

    def apply(self, stocks):
        filtered_stocks = stocks
        for filter in self.filters:
            filtered_stocks = filter.apply(filtered_stocks)
        return filtered_stocks

    def __str__(self):
        return f"Screen(screen_id={self.screen_id}, name='{self.name}', filters={self.filters})"

class SavedList:
    def __init__(self, list_id, name, stocks=None):
        self.list_id = list_id
        self.name = name
        self.stocks = stocks if stocks else []

    def add_stock(self, stock):
        self.stocks.append(stock)

    def remove_stock(self, stock):
        self.stocks.remove(stock)

    def __str__(self):
        return f"List(list_id={self.list_id}, name='{self.name}', stocks=[{', '.join([stock.symbol for stock in self.stocks])}])"

class Alert:
    def __init__(self, alert_id, stock, price_condition, price_value):
        self.alert_id = alert_id
        self.stock = stock
        self.price_condition = price_condition
        self.price_value = price_value

    def check_condition(self):
        if self.stock.price is None:
            return False
        if self.price_condition == "above" and self.stock.price > self.price_value:
            return True
        if self.price_condition == "below" and self.stock.price < self.price_value:
            return True
        return False

    def send_notification(self):
        print(f"Alert triggered for {self.stock.symbol}: Current price {self.stock.price} {self.price_condition} {self.price_value}")

    def __str__(self):
        return f"Alert(alert_id={self.alert_id}, stock='{self.stock.symbol}', price_condition='{self.price_condition}', price_value={self.price_value})"

class StockScreenApp:
    USERS_FILE = 'users.json'

    def __init__(self):
        if not os.path.exists(self.USERS_FILE):
            with open(self.USERS_FILE, 'w') as f:
                json.dump([], f) # Initialize with an empty JSON array
    
        self.users = {}
        self.screens = {}
        self.lists = {}
        self.alerts = {}
        self.user_id_counter = 1
        self.screen_id_counter = 1
        self.list_id_counter = 1
        self.alert_id_counter = 1
        self.current_user = None
        self.load_users()

    def load_users(self):
        if os.path.exists(self.USERS_FILE):
            if os.stat(self.USERS_FILE).st_size > 0:
                with open(self.USERS_FILE, 'r') as file:
                    user_dicts = json.load(file)
                    for user_dict in user_dicts:
                        user = User.from_dict(user_dict)
                        self.users[user.user_id] = user
                        if user.user_id >= self.user_id_counter:
                            self.user_id_counter = user.user_id + 1

    def save_users(self):
        with open(self.USERS_FILE, 'w') as file:
            json.dump([user.to_dict() for user in self.users.values()], file)

    def create_account(self, username, password):
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
        user = User(self.user_id_counter, username, hashed_password)
        self.users[self.user_id_counter] = user
        self.user_id_counter += 1
        self.save_users()
        return user

    def login(self, username, password):
        for user_id, user in self.users.items():
            if user.username == username and bcrypt.checkpw(password.encode('utf-8'), user.password_hash):
                self.current_user = user
                return user
        return None

    def logout(self):
        self.current_user = None

    def _load_stock_data(self, symbols):
        stock_data = {}
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                # data = stock.fast_info
                if stock:
                    stock_data[symbol] = StockData.from_yahoo_data(symbol, stock)
            except Exception as e:
                print(f"Could not get data for {symbol}: {e}")
        return stock_data

    def search(self, filters, symbols):
      stocks = list(self._load_stock_data(symbols).values())
      if len(stocks) == 0:
        return stocks
      
      filter_obj = Filter(filters[0], filters[1]) if filters else None
      if not filter_obj:
        return stocks
      return filter_obj.apply(stocks)

    def save_screen(self, screen):
      self.screens[screen.screen_id] = screen

    def load_screen(self, screen_id):
      return self.screens.get(screen_id)

    def save_list(self, saved_list):
       self.lists[saved_list.list_id] = saved_list

    def load_list(self, list_id):
      return self.lists.get(list_id)

    def create_screen(self, name):
        screen = StockScreen(self.screen_id_counter, name)
        self.screen_id_counter += 1
        return screen

    def create_list(self, name):
      list = SavedList(self.list_id_counter, name)
      self.list_id_counter += 1
      return list

    def set_alert(self, stock, price_condition, price_value):
      alert = Alert(self.alert_id_counter, stock, price_condition, price_value)
      self.alerts[self.alert_id_counter] = alert
      self.alert_id_counter += 1
      return alert

    def check_alerts(self):
      for alert_id, alert in self.alerts.items():
        stocks = list(self._load_stock_data([alert.stock.symbol]).values())
        if stocks and alert.stock.price != stocks[0].price:
           alert.stock = stocks[0]
        if alert.check_condition():
           alert.send_notification()


    def display_stock_data(self, stocks):
      if not stocks:
        print("No Stocks available")
        return

      for stock in stocks:
        print(stock)

    def display_menu(self):
        print("\nStock Screener App Menu:")
        print("1. Login")
        print("2. Create Account")
        print("3. Logout")
        print("4. Search Stocks")
        print("5. Create Screen")
        print("6. Load Screen and Apply")
        print("7. Create List")
        print("8. Load List")
        print("9. Set Alert")
        print("10. Check Alerts")
        print("11. Exit")

    def run(self):
        while True:
            self.display_menu()
            choice = input("Enter your choice: ")

            if choice == '1':
                username = input("Enter username: ")
                password = input("Enter password: ")
                user = self.login(username, password)
                if user:
                    print("Login successful")
                else:
                  print("Login Failed")
            elif choice == '2':
              username = input("Enter new username: ")
              password = input("Enter new password: ")
              user = self.create_account(username, password)
              print(f"Account Created: {user}")
            elif choice == '3':
                self.logout()
                print("Logged out")
            elif choice == '4' and self.current_user:
                symbols = input("Enter stock symbols separated by commas: ").split(',')
                filter_input = input("Enter filter criteria (e.g., 'pe_ratio', 'less than', 15  or press enter for no filter):")
                filters = filter_input.split(',') if filter_input else None
                stocks = self.search(filters, symbols)
                self.display_stock_data(stocks)
            elif choice == '5' and self.current_user:
                screen_name = input("Enter screen name: ")
                screen = self.create_screen(screen_name)
                while True:
                  filter_input = input("Enter filter criteria (e.g., 'pe_ratio', 'less than', 15  or type 'done' to finish), delimited by ',':")
                  if filter_input == "done":
                    break
                  if filter_input.strip():
                      filters = filter_input.split(',')
                      filter = Filter( {filters[0]: float(filters[2])}, filters[1])
                      screen.add_filter(filter)
                  else:
                      print("No filter criteria entered.")
                self.save_screen(screen)
                print("Screen saved.")
            elif choice == '6' and self.current_user:
              screen_id = int(input("Enter Screen ID(not name) to load: "))
              screen = self.load_screen(screen_id)
              if screen:
                 symbols = input("Enter stock symbols separated by commas: ").split(',')
                 stocks = self._load_stock_data(symbols)
                 filtered_stocks = screen.apply(list(stocks.values()))
                 self.display_stock_data(filtered_stocks)
              else:
                 print("Screen not found")

            elif choice == '7' and self.current_user:
              list_name = input("Enter list name: ")
              saved_list = self.create_list(list_name)
              while True:
                  symbol = input("Enter stock symbol to save to list  or type 'done' to finish: ")
                  if symbol == "done":
                    break
                  stocks = list(self._load_stock_data([symbol]).values())
                  if len(stocks):
                    saved_list.add_stock(stocks[0])
                  else:
                      print(f"Could not get stock data for {symbol}, skipping")
              self.save_list(saved_list)
              print("List saved")
            elif choice == '8' and self.current_user:
              list_id = int(input("Enter list id to load: "))
              loaded_list = self.load_list(list_id)
              if loaded_list:
                self.display_stock_data(loaded_list.stocks)
              else:
                print("List not found")
            elif choice == '9' and self.current_user:
              symbol = input("Enter symbol for alert: ")
              price_condition = input("Enter 'above' or 'below' condition: ")
              price_value = float(input("Enter price for alert trigger: "))
              stocks = list(self._load_stock_data([symbol]).values())
              if not stocks:
                 print("Could not get data for stock, alert not set")
                 continue
              alert = self.set_alert(stocks[0], price_condition, price_value)
              print(f"Alert set: {alert}")
            elif choice == '10' and self.current_user:
                self.check_alerts()
            elif choice == '11':
                print("Exiting")
                break
            elif not self.current_user and (choice != '1' and choice != '2'):
                print("Please log in first")
            else:
                print("Invalid choice")

if __name__ == "__main__":
    app = StockScreenApp()
    app.run()