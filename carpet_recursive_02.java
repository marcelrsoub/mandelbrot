package fractal_geometry;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.Random;

public class carpet_recursive_02 implements MouseListener {
	int x1,y1,length;
	int count=6;
	int c1,c2,c3;
	Graphics grph;
	public carpet_recursive_02(Graphics grph) {
		// TODO Auto-generated constructor stub
		this.grph=grph;
	}
	public void Initiation(MouseEvent e) {
		grph.clearRect(0, 0, 1000, 1000);
		Random rd=new Random();
		x1=rd.nextInt(500);
		y1=rd.nextInt(500);
		length=rd.nextInt(500);
		c1=rd.nextInt(256);
		c2=rd.nextInt(256);
		c3=rd.nextInt(256);
		this.draw(count,x1,y1,length);
		
	}
	private void draw(int count,int x1,int y1,int length) {
		int x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,length2;
		if(count>0) {
			count--;
			//画中间的方形
			length2=length/3;
			x2=x1+length2;
			y2=y1+length2;
			grph.setColor(new Color(c1,c2,c3));
			grph.fillRect(x1, y1, length, length);
			x3=x1;
			y3=y1+length2;
			x4=x1;
			y4=y1+length2;
			x5=x1+length2;
			y5=y1;
			x6=x1+length2;
			y6=y1+2*length2;
			x7=x1+2*length2;
			y7=y1;
			x8=x1+2*length2;
			y8=y1+length2;
			x9=x1+2*length2;
			y9=y1+2*length2;
			
			this.draw(count, x1, y1, length2);
			this.draw(count, x3, y3, length2);
			this.draw(count, x4, y4, length2);
			this.draw(count, x5, y5, length2);
			this.draw(count, x6, y6, length2);
			this.draw(count, x7, y7, length2);
			this.draw(count, x8, y8, length2);
			this.draw(count, x9, y9, length2);
		}else {
			return;
		}
	
		
	}
	
	
	
	
	
	public void mouseClicked(MouseEvent e) {
		 
	}
 
	public void mouseReleased(MouseEvent e) {
		
	}
 
	public void mouseEntered(MouseEvent e) {
		
	}
 
	public void mouseExited(MouseEvent e) {
		
	}

}
