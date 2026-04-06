-- Create and use erp schema
CREATE SCHEMA IF NOT EXISTS erp;
SET search_path TO erp;

-- Large ERP PostgreSQL schema and sample data
-- This script creates 22+ tables and populates them with sample data for testing/benchmarking

-- Drop tables if they exist
DROP TABLE IF EXISTS erp_expense_items, erp_expense_reports, erp_asset_assignments, erp_assets, erp_timesheets, erp_tasks, erp_projects, erp_purchase_order_items, erp_purchase_orders, erp_shipments, erp_inventory, erp_payments, erp_invoices, erp_suppliers, erp_order_items, erp_orders, erp_products, erp_customers, erp_employees, erp_departments, erp_user_roles, erp_roles, erp_permissions, erp_user_accounts CASCADE;

-- Departments
CREATE TABLE erp_departments (
    department_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

-- Employees
CREATE TABLE erp_employees (
    employee_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    department_id INT REFERENCES erp_departments(department_id),
    hire_date DATE,
    salary NUMERIC(12,2)
);

-- Customers
CREATE TABLE erp_customers (
    customer_id SERIAL PRIMARY KEY,
    company_name VARCHAR(100),
    contact_name VARCHAR(100),
    contact_email VARCHAR(100),
    address TEXT,
    city VARCHAR(50),
    country VARCHAR(50)
);

-- Products
CREATE TABLE erp_products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    description TEXT,
    price NUMERIC(10,2),
    stock INT
);

-- Orders
CREATE TABLE erp_orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES erp_customers(customer_id),
    employee_id INT REFERENCES erp_employees(employee_id),
    order_date DATE,
    status VARCHAR(30)
);

-- Order Items
CREATE TABLE erp_order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INT REFERENCES erp_orders(order_id),
    product_id INT REFERENCES erp_products(product_id),
    quantity INT,
    unit_price NUMERIC(10,2)
);

-- Suppliers
CREATE TABLE erp_suppliers (
    supplier_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    contact_name VARCHAR(100),
    contact_email VARCHAR(100),
    address TEXT,
    city VARCHAR(50),
    country VARCHAR(50)
);

-- Purchase Orders
CREATE TABLE erp_purchase_orders (
    po_id SERIAL PRIMARY KEY,
    supplier_id INT REFERENCES erp_suppliers(supplier_id),
    employee_id INT REFERENCES erp_employees(employee_id),
    po_date DATE,
    status VARCHAR(30)
);

-- Purchase Order Items
CREATE TABLE erp_purchase_order_items (
    po_item_id SERIAL PRIMARY KEY,
    po_id INT REFERENCES erp_purchase_orders(po_id),
    product_id INT REFERENCES erp_products(product_id),
    quantity INT,
    unit_price NUMERIC(10,2)
);

-- Invoices
CREATE TABLE erp_invoices (
    invoice_id SERIAL PRIMARY KEY,
    order_id INT REFERENCES erp_orders(order_id),
    invoice_date DATE,
    due_date DATE,
    total NUMERIC(12,2),
    status VARCHAR(30)
);

-- Payments
CREATE TABLE erp_payments (
    payment_id SERIAL PRIMARY KEY,
    invoice_id INT REFERENCES erp_invoices(invoice_id),
    payment_date DATE,
    amount NUMERIC(12,2),
    method VARCHAR(30)
);

-- Inventory
CREATE TABLE erp_inventory (
    inventory_id SERIAL PRIMARY KEY,
    product_id INT REFERENCES erp_products(product_id),
    warehouse VARCHAR(100),
    quantity INT
);

-- Shipments
CREATE TABLE erp_shipments (
    shipment_id SERIAL PRIMARY KEY,
    order_id INT REFERENCES erp_orders(order_id),
    shipped_date DATE,
    carrier VARCHAR(50),
    tracking_number VARCHAR(100)
);

-- Projects
CREATE TABLE erp_projects (
    project_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    description TEXT,
    start_date DATE,
    end_date DATE
);

-- Tasks
CREATE TABLE erp_tasks (
    task_id SERIAL PRIMARY KEY,
    project_id INT REFERENCES erp_projects(project_id),
    assigned_to INT REFERENCES erp_employees(employee_id),
    name VARCHAR(100),
    description TEXT,
    due_date DATE,
    status VARCHAR(30)
);

-- Timesheets
CREATE TABLE erp_timesheets (
    timesheet_id SERIAL PRIMARY KEY,
    employee_id INT REFERENCES erp_employees(employee_id),
    project_id INT REFERENCES erp_projects(project_id),
    date_worked DATE,
    hours NUMERIC(4,2)
);

-- Assets
CREATE TABLE erp_assets (
    asset_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    description TEXT,
    purchase_date DATE,
    value NUMERIC(12,2)
);

-- Asset Assignments
CREATE TABLE erp_asset_assignments (
    assignment_id SERIAL PRIMARY KEY,
    asset_id INT REFERENCES erp_assets(asset_id),
    employee_id INT REFERENCES erp_employees(employee_id),
    assigned_date DATE,
    return_date DATE
);

-- Expense Reports
CREATE TABLE erp_expense_reports (
    report_id SERIAL PRIMARY KEY,
    employee_id INT REFERENCES erp_employees(employee_id),
    report_date DATE,
    total NUMERIC(10,2),
    status VARCHAR(30)
);

-- Expense Items
CREATE TABLE erp_expense_items (
    item_id SERIAL PRIMARY KEY,
    report_id INT REFERENCES erp_expense_reports(report_id),
    description TEXT,
    amount NUMERIC(8,2)
);

-- User Accounts
CREATE TABLE erp_user_accounts (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE,
    password_hash VARCHAR(100),
    employee_id INT REFERENCES erp_employees(employee_id)
);

-- Roles
CREATE TABLE erp_roles (
    role_id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE
);

-- Permissions
CREATE TABLE erp_permissions (
    permission_id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE
);

-- User Roles
CREATE TABLE erp_user_roles (
    user_id INT REFERENCES erp_user_accounts(user_id),
    role_id INT REFERENCES erp_roles(role_id),
    PRIMARY KEY (user_id, role_id)
);

-- Insert sample data for each table
-- Departments
INSERT INTO erp_departments (name) VALUES
('Sales'), ('HR'), ('IT'), ('Finance'), ('Logistics'), ('Procurement'), ('R&D'), ('Support'), ('Marketing'), ('Legal');

-- Employees (100)
INSERT INTO erp_employees (first_name, last_name, department_id, hire_date, salary)
SELECT 'Emp' || g, 'Test', (random()*9+1)::int, date '2015-01-01' + (random()*3650)::int, (random()*50000+40000)::numeric(12,2)
FROM generate_series(1,100) g;

-- Customers (100)
INSERT INTO erp_customers (company_name, contact_name, contact_email, address, city, country)
SELECT 'Company ' || g, 'Contact ' || g, 'contact' || g || '@company.com', 'Addr ' || g, 'City ' || (g%10+1), 'Country ' || (g%5+1)
FROM generate_series(1,100) g;

-- Products (200)
INSERT INTO erp_products (name, description, price, stock)
SELECT 'Product ' || g, 'Desc ' || g, (random()*100+10)::numeric(10,2), (random()*1000)::int
FROM generate_series(1,200) g;

-- Orders (2000)
INSERT INTO erp_orders (customer_id, employee_id, order_date, status)
SELECT (random()*99+1)::int, (random()*99+1)::int, date '2024-01-01' + (random()*120)::int, CASE WHEN random() < 0.7 THEN 'Completed' ELSE 'Pending' END
FROM generate_series(1,2000);

-- Order Items (10000)
INSERT INTO erp_order_items (order_id, product_id, quantity, unit_price)
SELECT (random()*1999+1)::int, (random()*199+1)::int, (random()*20+1)::int, (random()*100+10)::numeric(10,2)
FROM generate_series(1,10000);

-- Suppliers (50)
INSERT INTO erp_suppliers (name, contact_name, contact_email, address, city, country)
SELECT 'Supplier ' || g, 'Contact ' || g, 'contact' || g || '@supplier.com', 'Addr ' || g, 'City ' || (g%10+1), 'Country ' || (g%5+1)
FROM generate_series(1,50) g;

-- Purchase Orders (500)
INSERT INTO erp_purchase_orders (supplier_id, employee_id, po_date, status)
SELECT (random()*49+1)::int, (random()*99+1)::int, date '2024-01-01' + (random()*120)::int, CASE WHEN random() < 0.8 THEN 'Approved' ELSE 'Pending' END
FROM generate_series(1,500);

-- Purchase Order Items (2000)
INSERT INTO erp_purchase_order_items (po_id, product_id, quantity, unit_price)
SELECT (random()*499+1)::int, (random()*199+1)::int, (random()*50+1)::int, (random()*100+10)::numeric(10,2)
FROM generate_series(1,2000);

-- Invoices (1500)
INSERT INTO erp_invoices (order_id, invoice_date, due_date, total, status)
SELECT (random()*1999+1)::int, date '2024-01-01' + (random()*120)::int, date '2024-05-01' + (random()*60)::int, (random()*1000+100)::numeric(12,2), CASE WHEN random() < 0.8 THEN 'Paid' ELSE 'Unpaid' END
FROM generate_series(1,1500);

-- Payments (1200)
INSERT INTO erp_payments (invoice_id, payment_date, amount, method)
SELECT (random()*1499+1)::int, date '2024-01-01' + (random()*180)::int, (random()*1000+100)::numeric(12,2), CASE WHEN random() < 0.5 THEN 'Bank' ELSE 'Card' END
FROM generate_series(1,1200);

-- Inventory (200)
INSERT INTO erp_inventory (product_id, warehouse, quantity)
SELECT g, 'Warehouse ' || (g%5+1), (random()*1000)::int FROM generate_series(1,200) g;

-- Shipments (800)
INSERT INTO erp_shipments (order_id, shipped_date, carrier, tracking_number)
SELECT (random()*1999+1)::int, date '2024-01-01' + (random()*120)::int, 'Carrier ' || (g%5+1), 'TRK' || g FROM generate_series(1,800) g;

-- Projects (30)
INSERT INTO erp_projects (name, description, start_date, end_date)
SELECT 'Project ' || g, 'Desc ' || g, date '2023-01-01' + (random()*365)::int, date '2024-01-01' + (random()*365)::int FROM generate_series(1,30) g;

-- Tasks (300)
INSERT INTO erp_tasks (project_id, assigned_to, name, description, due_date, status)
SELECT (random()*29+1)::int, (random()*99+1)::int, 'Task ' || g, 'Desc ' || g, date '2024-01-01' + (random()*120)::int, CASE WHEN random() < 0.7 THEN 'Open' ELSE 'Closed' END FROM generate_series(1,300) g;

-- Timesheets (2000)
INSERT INTO erp_timesheets (employee_id, project_id, date_worked, hours)
SELECT (random()*99+1)::int, (random()*29+1)::int, date '2024-01-01' + (random()*120)::int, (random()*8+1)::numeric(4,2) FROM generate_series(1,2000);

-- Assets (100)
INSERT INTO erp_assets (name, description, purchase_date, value)
SELECT 'Asset ' || g, 'Desc ' || g, date '2020-01-01' + (random()*1500)::int, (random()*10000+1000)::numeric(12,2) FROM generate_series(1,100) g;

-- Asset Assignments (150)
INSERT INTO erp_asset_assignments (asset_id, employee_id, assigned_date, return_date)
SELECT (random()*99+1)::int, (random()*99+1)::int, date '2024-01-01' + (random()*120)::int, NULL FROM generate_series(1,150);

-- Expense Reports (300)
INSERT INTO erp_expense_reports (employee_id, report_date, total, status)
SELECT (random()*99+1)::int, date '2024-01-01' + (random()*120)::int, (random()*2000+100)::numeric(10,2), CASE WHEN random() < 0.7 THEN 'Approved' ELSE 'Pending' END FROM generate_series(1,300);

-- Expense Items (900)
INSERT INTO erp_expense_items (report_id, description, amount)
SELECT (random()*299+1)::int, 'Expense ' || g, (random()*500+10)::numeric(8,2) FROM generate_series(1,900) g;

-- User Accounts (100)
INSERT INTO erp_user_accounts (username, password_hash, employee_id)
SELECT 'user' || g, 'hash' || g, g FROM generate_series(1,100) g;

-- Roles (5)
INSERT INTO erp_roles (name) VALUES ('Admin'), ('Manager'), ('User'), ('Auditor'), ('Guest');

-- Permissions (10)
INSERT INTO erp_permissions (name) VALUES ('read'), ('write'), ('delete'), ('approve'), ('export'), ('import'), ('manage_users'), ('manage_roles'), ('view_reports'), ('edit_settings');

-- User Roles (100)
INSERT INTO erp_user_roles (user_id, role_id)
SELECT g, (random()*4+1)::int FROM generate_series(1,100) g;

-- Indexes for performance
CREATE INDEX idx_erp_orders_customer_id ON erp_orders(customer_id);
CREATE INDEX idx_erp_orders_employee_id ON erp_orders(employee_id);
CREATE INDEX idx_erp_order_items_order_id ON erp_order_items(order_id);
CREATE INDEX idx_erp_order_items_product_id ON erp_order_items(product_id);
CREATE INDEX idx_erp_purchase_orders_supplier_id ON erp_purchase_orders(supplier_id);
CREATE INDEX idx_erp_purchase_orders_employee_id ON erp_purchase_orders(employee_id);
CREATE INDEX idx_erp_purchase_order_items_po_id ON erp_purchase_order_items(po_id);
CREATE INDEX idx_erp_purchase_order_items_product_id ON erp_purchase_order_items(product_id);
CREATE INDEX idx_erp_invoices_order_id ON erp_invoices(order_id);
CREATE INDEX idx_erp_payments_invoice_id ON erp_payments(invoice_id);
CREATE INDEX idx_erp_inventory_product_id ON erp_inventory(product_id);
CREATE INDEX idx_erp_shipments_order_id ON erp_shipments(order_id);
CREATE INDEX idx_erp_tasks_project_id ON erp_tasks(project_id);
CREATE INDEX idx_erp_tasks_assigned_to ON erp_tasks(assigned_to);
CREATE INDEX idx_erp_timesheets_employee_id ON erp_timesheets(employee_id);
CREATE INDEX idx_erp_timesheets_project_id ON erp_timesheets(project_id);
CREATE INDEX idx_erp_asset_assignments_asset_id ON erp_asset_assignments(asset_id);
CREATE INDEX idx_erp_asset_assignments_employee_id ON erp_asset_assignments(employee_id);
CREATE INDEX idx_erp_expense_reports_employee_id ON erp_expense_reports(employee_id);
CREATE INDEX idx_erp_expense_items_report_id ON erp_expense_items(report_id);
CREATE INDEX idx_erp_user_accounts_employee_id ON erp_user_accounts(employee_id);
CREATE INDEX idx_erp_user_roles_user_id ON erp_user_roles(user_id);
CREATE INDEX idx_erp_user_roles_role_id ON erp_user_roles(role_id);
