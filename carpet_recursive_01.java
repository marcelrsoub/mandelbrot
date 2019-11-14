package fractal_geometry;
import java.awt.Graphics;
import java.awt.event.MouseEvent;

import javax.swing.JFrame;

import org.omg.CORBA.PRIVATE_MEMBER;
import org.omg.CORBA.PUBLIC_MEMBER;

public class carpet_recursive_01 extends JFrame{

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		carpet_recursive_01 carpet=new carpet_recursive_01();
		carpet.initUI();	
	}
	private void initUI() {
		this.setName("Sierpinski Carpet");
		this.setSize(500,500);
		//点击关闭关闭窗口
		this.setDefaultCloseOperation(3);
		//此窗口将置于屏幕的中央
		this.setLocationRelativeTo(null);
		this.setVisible(true);
		Graphics grph=this.getGraphics();
		carpet_recursive_02 draw=new carpet_recursive_02(grph);
		this.addMouseListener(draw);
		
	}

}
